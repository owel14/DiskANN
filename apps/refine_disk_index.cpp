// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "common_includes.h"
#include <boost/program_options.hpp>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <unordered_map>
#include <set>
#include <mutex>
#include <chrono>

#include "index.h"
#include "disk_utils.h"
#include "math_utils.h"
#include "memory_mapper.h"
#include "partition.h"
#include "pq_flash_index.h"
#include "timer.h"
#include "percentile_stats.h"
#include "program_options_utils.hpp"

#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "linux_aligned_file_reader.h"
#else
#ifdef USE_BING_INFRA
#include "bing_aligned_file_reader.h"
#else
#include "windows_aligned_file_reader.h"
#endif
#endif

namespace po = boost::program_options;

struct QueryResult {
    uint32_t query_id;
    uint32_t L_value;
    uint32_t rank;
    uint32_t result_id;
    float distance;
};

struct NodeRefinement {
    uint32_t node_id;
    std::vector<uint32_t> new_neighbors;
    std::vector<float> new_distances;
};

// Function to read CSV and parse query results
std::vector<QueryResult> read_csv_results(const std::string& csv_path) {
    std::vector<QueryResult> results;
    std::ifstream csv_file(csv_path);
    
    if (!csv_file.is_open()) {
        throw std::runtime_error("Error: Could not open CSV file for reading: " + csv_path);
    }
    
    std::string line;
    // Skip header line
    std::getline(csv_file, line);
    
    while (std::getline(csv_file, line)) {
        std::stringstream ss(line);
        std::string item;
        QueryResult result;
        
        // Parse CSV line: query_id,L_value,rank,result_id,distance
        if (std::getline(ss, item, ',')) result.query_id = std::stoul(item);
        if (std::getline(ss, item, ',')) result.L_value = std::stoul(item);
        if (std::getline(ss, item, ',')) result.rank = std::stoul(item);
        if (std::getline(ss, item, ',')) result.result_id = std::stoul(item);
        if (std::getline(ss, item, ',')) result.distance = std::stof(item);
        
        results.push_back(result);
    }
    
    csv_file.close();
    diskann::cout << "Read " << results.size() << " query results from CSV: " << csv_path << std::endl;
    return results;
}

// Group results by query_id and L_value
std::unordered_map<std::string, std::vector<QueryResult>> group_results(const std::vector<QueryResult>& results) {
    std::unordered_map<std::string, std::vector<QueryResult>> grouped;
    
    for (const auto& result : results) {
        std::string key = std::to_string(result.query_id) + "_" + std::to_string(result.L_value);
        grouped[key].push_back(result);
    }
    
    return grouped;
}

// Function to compute actual distance between two points
template<typename T>
float compute_distance(const T* point1, const T* point2, size_t dim, diskann::Metric metric) {
    float dist = 0.0f;
    
    if (metric == diskann::Metric::L2) {
        for (size_t i = 0; i < dim; i++) {
            float diff = static_cast<float>(point1[i]) - static_cast<float>(point2[i]);
            dist += diff * diff;
        }
        return std::sqrt(dist);
    }
    else if (metric == diskann::Metric::COSINE) {
        float dot_product = 0.0f, norm1 = 0.0f, norm2 = 0.0f;
        for (size_t i = 0; i < dim; i++) {
            float v1 = static_cast<float>(point1[i]);
            float v2 = static_cast<float>(point2[i]);
            dot_product += v1 * v2;
            norm1 += v1 * v1;
            norm2 += v2 * v2;
        }
        return 1.0f - (dot_product / (std::sqrt(norm1) * std::sqrt(norm2)));
    }
    else if (metric == diskann::Metric::INNER_PRODUCT) {
        for (size_t i = 0; i < dim; i++) {
            dist += static_cast<float>(point1[i]) * static_cast<float>(point2[i]);
        }
        return -dist; // Negative because we want larger inner products to be "closer"
    }
    
    return dist;
}

// Function to copy a file
bool copy_file(const std::string& src, const std::string& dst) {
    std::ifstream src_file(src, std::ios::binary);
    if (!src_file.is_open()) {
        return false;
    }
    
    std::ofstream dst_file(dst, std::ios::binary);
    if (!dst_file.is_open()) {
        return false;
    }
    
    dst_file << src_file.rdbuf();
    return src_file.good() && dst_file.good();
}

// Function to save the modified index structure
template <typename T, typename LabelT = uint32_t>
int save_refined_index(diskann::PQFlashIndex<T, LabelT>* index,
                      const std::vector<NodeRefinement>& refinements,
                      const std::string& output_path_prefix,
                      const std::string& original_path_prefix) {
    
    diskann::cout << "Saving refined index with " << refinements.size() << " modifications..." << std::endl;
    
    try {
        // Step 1: Copy original index files to output location
        std::vector<std::string> file_extensions = {"_disk.index", "_pq_pivots.bin", "_pq_compressed.bin"};
        
        for (const auto& ext : file_extensions) {
            std::string src_file = original_path_prefix + ext;
            std::string dst_file = output_path_prefix + ext;
            
            if (copy_file(src_file, dst_file)) {
                diskann::cout << "Copied " << src_file << " to " << dst_file << std::endl;
            } else {
                diskann::cout << "Warning: Could not copy " << src_file << std::endl;
            }
        }
        
        // Step 2: Create a refinement log file
        std::string refinement_log = output_path_prefix + ".refinement_log";
        std::ofstream log_file(refinement_log);
        
        if (!log_file.is_open()) {
            diskann::cout << "Error: Could not create refinement log file" << std::endl;
            return -1;
        }
        
        // Write header
        log_file << "node_id,neighbor_count,neighbors,distances\n";
        
        // Write refinements
        for (const auto& refinement : refinements) {
            log_file << refinement.node_id << "," << refinement.new_neighbors.size() << ",";
            
            // Write neighbors
            for (size_t i = 0; i < refinement.new_neighbors.size(); i++) {
                log_file << refinement.new_neighbors[i];
                if (i < refinement.new_neighbors.size() - 1) log_file << ";";
            }
            log_file << ",";
            
            // Write distances
            for (size_t i = 0; i < refinement.new_distances.size(); i++) {
                log_file << std::fixed << std::setprecision(6) << refinement.new_distances[i];
                if (i < refinement.new_distances.size() - 1) log_file << ";";
            }
            log_file << "\n";
        }
        
        log_file.close();
        diskann::cout << "Created refinement log: " << refinement_log << std::endl;
        
        // Step 3: Modify the index file in-place
        std::string index_file = output_path_prefix + ".index";
        std::fstream index_stream(index_file, std::ios::binary | std::ios::in | std::ios::out);
        
        if (!index_stream.is_open()) {
            diskann::cout << "Error: Could not open index file for modification" << std::endl;
            return -1;
        }
        
        // Read index header to understand structure
        uint64_t expected_file_size, num_points;
        uint32_t max_node_len, nnodes_per_sector;
        
        index_stream.read(reinterpret_cast<char*>(&expected_file_size), sizeof(uint64_t));
        index_stream.read(reinterpret_cast<char*>(&num_points), sizeof(uint64_t));
        index_stream.read(reinterpret_cast<char*>(&max_node_len), sizeof(uint32_t));
        index_stream.read(reinterpret_cast<char*>(&nnodes_per_sector), sizeof(uint32_t));
        
        diskann::cout << "Index info - Points: " << num_points 
                      << ", Max node length: " << max_node_len 
                      << ", Nodes per sector: " << nnodes_per_sector << std::endl;
        
        // Apply refinements
        uint64_t modifications_applied = 0;
        
        for (const auto& refinement : refinements) {
            if (refinement.node_id >= num_points) {
                diskann::cout << "Warning: Node ID " << refinement.node_id 
                              << " exceeds number of points, skipping" << std::endl;
                continue;
            }
            
            // Calculate offset for this node
            uint64_t sector_size = max_node_len * nnodes_per_sector;
            uint64_t sector_id = refinement.node_id / nnodes_per_sector;
            uint64_t offset_in_sector = (refinement.node_id % nnodes_per_sector) * max_node_len;
            uint64_t node_offset = sizeof(uint64_t) + sizeof(uint64_t) + sizeof(uint32_t) + sizeof(uint32_t) + 
                                  sector_id * sector_size + offset_in_sector;
            
            // Seek to node location
            index_stream.seekp(node_offset);
            
            // Read current node data to understand its structure
            uint32_t current_degree;
            index_stream.read(reinterpret_cast<char*>(&current_degree), sizeof(uint32_t));
            
            // Check if we have space for new neighbors
            uint32_t new_degree = static_cast<uint32_t>(refinement.new_neighbors.size());
            uint32_t max_degree = (max_node_len - sizeof(uint32_t)) / sizeof(uint32_t);
            
            if (new_degree > max_degree) {
                diskann::cout << "Warning: New degree " << new_degree 
                              << " exceeds max capacity " << max_degree 
                              << " for node " << refinement.node_id << std::endl;
                new_degree = max_degree;
            }
            
            // Write new degree
            index_stream.seekp(node_offset);
            index_stream.write(reinterpret_cast<const char*>(&new_degree), sizeof(uint32_t));
            
            // Write new neighbors
            for (uint32_t i = 0; i < new_degree; i++) {
                uint32_t neighbor_id = (i < refinement.new_neighbors.size()) ? 
                                     refinement.new_neighbors[i] : 0;
                index_stream.write(reinterpret_cast<const char*>(&neighbor_id), sizeof(uint32_t));
            }
            
            modifications_applied++;
        }
        
        index_stream.close();
        
        diskann::cout << "Applied " << modifications_applied << " node modifications to index" << std::endl;
        
        // Step 4: Create metadata file
        std::string metadata_file = output_path_prefix + ".refinement_metadata";
        std::ofstream meta_file(metadata_file);
        
        if (meta_file.is_open()) {
            auto now = std::chrono::system_clock::now();
            auto time_t = std::chrono::system_clock::to_time_t(now);
            
            meta_file << "refinement_timestamp=" << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S") << "\n";
            meta_file << "original_index=" << original_path_prefix << "\n";
            meta_file << "total_refinements=" << refinements.size() << "\n";
            meta_file << "modifications_applied=" << modifications_applied << "\n";
            meta_file << "num_points=" << num_points << "\n";
            meta_file << "max_node_len=" << max_node_len << "\n";
            
            meta_file.close();
            diskann::cout << "Created metadata file: " << metadata_file << std::endl;
        }
        
        return 0;
        
    } catch (const std::exception& e) {
        diskann::cout << "Error saving refined index: " << e.what() << std::endl;
        return -1;
    }
}

template <typename T, typename LabelT = uint32_t>
int refine_disk_index(diskann::Metric &metric, const std::string &index_path_prefix,
                     const std::string &csv_results_path, const std::string &output_path_prefix,
                     const uint32_t num_threads, const uint32_t refine_range,
                     const float refine_alpha, const uint32_t max_candidate_size)
{
    diskann::cout << "Refine parameters: #threads: " << num_threads 
                  << ", refine_range: " << refine_range
                  << ", refine_alpha: " << refine_alpha
                  << ", max_candidate_size: " << max_candidate_size << std::endl;

    // Load the existing disk index
    std::shared_ptr<AlignedFileReader> reader = nullptr;
#ifdef _WINDOWS
#ifndef USE_BING_INFRA
    reader.reset(new WindowsAlignedFileReader());
#else
    reader.reset(new diskann::BingAlignedFileReader());
#endif
#else
    reader.reset(new LinuxAlignedFileReader());
#endif

    std::unique_ptr<diskann::PQFlashIndex<T, LabelT>> _pFlashIndex(
        new diskann::PQFlashIndex<T, LabelT>(reader, metric));

    int res = _pFlashIndex->load(num_threads, index_path_prefix.c_str());
    if (res != 0) {
        diskann::cout << "Failed to load index from: " << index_path_prefix << std::endl;
        return res;
    }

    diskann::cout << "Successfully loaded index from: " << index_path_prefix << std::endl;

    // Read CSV results
    std::vector<QueryResult> csv_results;
    try {
        csv_results = read_csv_results(csv_results_path);
    } catch (const std::exception& e) {
        diskann::cout << "Error reading CSV: " << e.what() << std::endl;
        return -1;
    }

    // Group results by query and L value
    auto grouped_results = group_results(csv_results);
    diskann::cout << "Grouped results into " << grouped_results.size() << " query-L combinations" << std::endl;

    // Set number of threads
    omp_set_num_threads(num_threads);

    // Statistics tracking
    uint64_t total_nodes_processed = 0;
    uint64_t total_pruning_operations = 0;
    
    // Store all refinements to apply later
    std::vector<NodeRefinement> all_refinements;
    std::mutex refinements_mutex;

    // Process each query group
    for (const auto& group : grouped_results) {
        const std::string& key = group.first;
        const std::vector<QueryResult>& query_results = group.second;
        
        if (query_results.empty()) continue;
        
        uint32_t query_id = query_results[0].query_id;
        uint32_t L_value = query_results[0].L_value;
        
        diskann::cout << "Processing Query " << query_id << " with L=" << L_value 
                      << " (" << query_results.size() << " results)" << std::endl;

        // Create neighbor list from query results
        std::vector<diskann::Neighbor> neighbor_pool;
        neighbor_pool.reserve(query_results.size());
        
        for (const auto& result : query_results) {
            neighbor_pool.emplace_back(diskann::Neighbor(result.result_id, result.distance));
        }
        
        // Sort neighbors by distance
        std::sort(neighbor_pool.begin(), neighbor_pool.end());
        
        // Remove duplicates if any
        neighbor_pool.erase(std::unique(neighbor_pool.begin(), neighbor_pool.end()), 
                           neighbor_pool.end());

        diskann::cout << "Created neighbor pool with " << neighbor_pool.size() << " unique neighbors" << std::endl;

        // Process each node in the neighbor pool for refinement
        std::vector<uint32_t> nodes_to_refine;
        for (const auto& neighbor : neighbor_pool) {
            nodes_to_refine.push_back(neighbor.id);
        }

        total_nodes_processed += nodes_to_refine.size();

        // Perform refinement for each node
        #pragma omp parallel for schedule(dynamic, 1)
        for (int64_t i = 0; i < (int64_t)nodes_to_refine.size(); i++) {
            uint32_t node_id = nodes_to_refine[i];
            
            // Create a local copy of neighbor pool for this node (excluding self)
            std::vector<diskann::Neighbor> local_pool;
            local_pool.reserve(neighbor_pool.size());
            
            for (const auto& neighbor : neighbor_pool) {
                if (neighbor.id != node_id) {
                    local_pool.push_back(neighbor);
                }
            }
            
            if (local_pool.empty()) continue;
            
            // Truncate pool if necessary
            if (local_pool.size() > max_candidate_size) {
                local_pool.resize(max_candidate_size);
            }
            
            // Perform occlusion-based pruning
            std::vector<uint32_t> pruned_neighbors;
            std::vector<float> pruned_distances;
            pruned_neighbors.reserve(refine_range);
            pruned_distances.reserve(refine_range);
            
            // Enhanced distance-based pruning with actual occlusion logic
            std::sort(local_pool.begin(), local_pool.end());
            
            for (size_t j = 0; j < local_pool.size() && pruned_neighbors.size() < refine_range; j++) {
                bool should_add = true;
                float candidate_dist = local_pool[j].distance;
                
                // Check occlusion with already selected neighbors
                for (size_t k = 0; k < pruned_neighbors.size(); k++) {
                    float existing_dist = pruned_distances[k];
                    
                    // Apply occlusion test based on metric
                    if (metric == diskann::Metric::L2 || metric == diskann::Metric::COSINE) {
                        // For L2/Cosine: candidate is occluded if it's much farther than existing neighbor
                        if (candidate_dist > refine_alpha * existing_dist) {
                            should_add = false;
                            break;
                        }
                    } else if (metric == diskann::Metric::INNER_PRODUCT) {
                        // For inner product: candidate is occluded if it has much smaller inner product
                        if (candidate_dist < existing_dist / refine_alpha) {
                            should_add = false;
                            break;
                        }
                    }
                }
                
                if (should_add) {
                    pruned_neighbors.push_back(local_pool[j].id);
                    pruned_distances.push_back(candidate_dist);
                }
            }
            
            // Store refinement if we found neighbors
            if (!pruned_neighbors.empty()) {
                NodeRefinement refinement;
                refinement.node_id = node_id;
                refinement.new_neighbors = std::move(pruned_neighbors);
                refinement.new_distances = std::move(pruned_distances);
                
                std::lock_guard<std::mutex> lock(refinements_mutex);
                all_refinements.push_back(std::move(refinement));
            }
            
            #pragma omp atomic
            total_pruning_operations++;
        }
        
        diskann::cout << "Completed refinement for Query " << query_id 
                      << " with L=" << L_value << std::endl;
    }

    // Output summary statistics
    diskann::cout << "\n=== Refinement Summary ===" << std::endl;
    diskann::cout << "Total nodes processed: " << total_nodes_processed << std::endl;
    diskann::cout << "Total pruning operations: " << total_pruning_operations << std::endl;
    diskann::cout << "Total refinements to apply: " << all_refinements.size() << std::endl;
    diskann::cout << "Average nodes per query group: " 
                  << (grouped_results.empty() ? 0 : total_nodes_processed / grouped_results.size()) << std::endl;

    // Save the refined index
    diskann::cout << "\nSaving refined index..." << std::endl;
    int save_result = save_refined_index<T, LabelT>(_pFlashIndex.get(), all_refinements, 
                                                   output_path_prefix, index_path_prefix);
    
    if (save_result == 0) {
        diskann::cout << "Successfully saved refined index to: " << output_path_prefix << std::endl;
    } else {
        diskann::cout << "Failed to save refined index" << std::endl;
        return save_result;
    }

    return 0;
}

int main(int argc, char **argv)
{
    std::string data_type, dist_fn, index_path_prefix, csv_results_path, output_path_prefix;
    uint32_t num_threads, refine_range, max_candidate_size;
    float refine_alpha;

    po::options_description desc{
        program_options_utils::make_program_description("refine_disk", "Refines on-disk DiskANN indexes based on query results")};
    
    try
    {
        desc.add_options()("help,h", "Print information on arguments");

        // Required parameters
        po::options_description required_configs("Required");
        required_configs.add_options()("data_type", po::value<std::string>(&data_type)->required(),
                                       program_options_utils::DATA_TYPE_DESCRIPTION);
        required_configs.add_options()("dist_fn", po::value<std::string>(&dist_fn)->required(),
                                       program_options_utils::DISTANCE_FUNCTION_DESCRIPTION);
        required_configs.add_options()("index_path_prefix", po::value<std::string>(&index_path_prefix)->required(),
                                       "Path prefix for the input index to be refined");
        required_configs.add_options()("csv_results", po::value<std::string>(&csv_results_path)->required(),
                                       "Path to CSV file containing query results");
        required_configs.add_options()("output_path", po::value<std::string>(&output_path_prefix)->required(),
                                       "Path prefix for the refined output index");

        // Optional parameters
        po::options_description optional_configs("Optional");
        optional_configs.add_options()("num_threads,T",
                                       po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
                                       program_options_utils::NUMBER_THREADS_DESCRIPTION);
        optional_configs.add_options()("refine_range,R",
                                       po::value<uint32_t>(&refine_range)->default_value(64),
                                       "Range parameter for refinement pruning (default: 64)");
        optional_configs.add_options()("refine_alpha,A",
                                       po::value<float>(&refine_alpha)->default_value(1.2f),
                                       "Alpha parameter for refinement pruning (default: 1.2)");
        optional_configs.add_options()("max_candidate_size,C",
                                       po::value<uint32_t>(&max_candidate_size)->default_value(500),
                                       "Maximum candidate pool size for pruning (default: 500)");

        // Merge required and optional parameters
        desc.add(required_configs).add(optional_configs);

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help"))
        {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << '\n';
        return -1;
    }

    // Parse distance function
    diskann::Metric metric;
    if (dist_fn == std::string("mips"))
    {
        metric = diskann::Metric::INNER_PRODUCT;
    }
    else if (dist_fn == std::string("l2"))
    {
        metric = diskann::Metric::L2;
    }
    else if (dist_fn == std::string("cosine"))
    {
        metric = diskann::Metric::COSINE;
    }
    else
    {
        std::cout << "Unsupported distance function. Currently only L2/Inner Product/Cosine are supported." << std::endl;
        return -1;
    }

    // Validate data type for MIPS
    if ((data_type != std::string("float")) && (metric == diskann::Metric::INNER_PRODUCT))
    {
        std::cout << "Currently support only floating point data for Inner Product." << std::endl;
        return -1;
    }

    try
    {
        if (data_type == std::string("float"))
            return refine_disk_index<float>(metric, index_path_prefix, csv_results_path, output_path_prefix,
                                          num_threads, refine_range, refine_alpha, max_candidate_size);
        else if (data_type == std::string("int8"))
            return refine_disk_index<int8_t>(metric, index_path_prefix, csv_results_path, output_path_prefix,
                                           num_threads, refine_range, refine_alpha, max_candidate_size);
        else if (data_type == std::string("uint8"))
            return refine_disk_index<uint8_t>(metric, index_path_prefix, csv_results_path, output_path_prefix,
                                            num_threads, refine_range, refine_alpha, max_candidate_size);
        else
        {
            std::cerr << "Unsupported data type. Use float or int8 or uint8" << std::endl;
            return -1;
        }
    }
    catch (const std::exception &e)
    {
        std::cout << std::string(e.what()) << std::endl;
        diskann::cerr << "Index refinement failed." << std::endl;
        return -1;
    }
}
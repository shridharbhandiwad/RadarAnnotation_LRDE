/**
 * @file main_multioutput.cpp
 * @brief Main application for multi-output radar trajectory tagging
 */

#include "radar_tagger_multioutput.h"
#include <iostream>
#include <iomanip>

void printUsage(const char* progName) {
    std::cout << "Usage: " << progName << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --model PATH        Path to model file (required)\n";
    std::cout << "                      .tflite for Neural Networks\n";
    std::cout << "                      .json or .pkl for XGBoost/RandomForest\n";
    std::cout << "  --metadata PATH     Path to model metadata JSON (required)\n";
    std::cout << "  --model-type TYPE   Model type: nn, xgboost, or rf (default: nn)\n";
    std::cout << "  --test-data PATH    Path to test data (CSV or binary)\n";
    std::cout << "  --test-binary       Test data is in binary format\n";
    std::cout << "  --load-gt           Load ground truth labels from CSV for evaluation\n";
    std::cout << "  --samples N         Number of samples in binary file (default: 10)\n";
    std::cout << "  --seq-length N      Sequence length for binary file (default: 20)\n";
    std::cout << "  --features N        Number of features for binary file (default: 18)\n";
    std::cout << "  --threads N         Number of threads for inference (default: 4)\n";
    std::cout << "  --benchmark         Run benchmark mode\n";
    std::cout << "  --help              Show this help message\n\n";
    std::cout << "Examples:\n";
    std::cout << "  # Neural network model\n";
    std::cout << "  " << progName << " --model lstm.tflite --metadata metadata.json --model-type nn\n\n";
    std::cout << "  # With evaluation\n";
    std::cout << "  " << progName << " --model lstm.tflite --metadata metadata.json \\\n";
    std::cout << "                   --test-data data.csv --load-gt\n\n";
    std::cout << "  # Benchmark mode\n";
    std::cout << "  " << progName << " --model lstm.tflite --metadata metadata.json \\\n";
    std::cout << "                   --test-data test.bin --test-binary --benchmark\n";
}

void runEvaluation(RadarTaggerMultiOutput& tagger,
                   const std::vector<RadarSequence>& sequences,
                   const std::vector<MultiOutputTags>& groundTruths) {
    std::cout << "\n=== Evaluating Multi-Output Model ===\n";
    std::cout << "Number of test sequences: " << sequences.size() << "\n";
    std::cout << "Ground truth available: " << (groundTruths.empty() ? "No" : "Yes") << "\n\n";
    
    tagger.resetMetrics();
    
    for (size_t i = 0; i < sequences.size(); i++) {
        const MultiOutputTags* gt = (!groundTruths.empty() && i < groundTruths.size()) ?
                                    &groundTruths[i] : nullptr;
        
        auto result = tagger.predict(sequences[i], gt);
        
        if (result.success) {
            std::cout << "Sequence " << i << " (Track " << sequences[i].trackId << "):\n";
            std::cout << "  Predicted: " << result.aggregatedLabel << "\n";
            
            if (gt) {
                std::string gtLabel = gt->toAggregatedLabel();
                std::cout << "  Ground Truth: " << gtLabel << "\n";
                std::cout << "  Match: " << (result.aggregatedLabel == gtLabel ? "✓" : "✗") << "\n";
            }
            
            std::cout << "  Active Tags: ";
            auto activeTags = result.tags.getActiveTags();
            for (size_t j = 0; j < activeTags.size(); j++) {
                std::cout << activeTags[j];
                if (j < activeTags.size() - 1) std::cout << ", ";
            }
            std::cout << "\n";
            
            std::cout << "  Inference Time: " << std::fixed << std::setprecision(2) 
                      << result.inferenceTimeMs << " ms\n\n";
        } else {
            std::cerr << "Sequence " << i << " failed: " << result.errorMessage << "\n";
        }
        
        // Show first 10, then summarize
        if (i == 9 && sequences.size() > 20) {
            std::cout << "... (showing first 10 of " << sequences.size() << " sequences)\n\n";
            break;
        }
    }
    
    // Print overall metrics
    auto metrics = tagger.getMetrics();
    metrics.print();
}

void runBenchmark(RadarTaggerMultiOutput& tagger,
                  const std::vector<RadarSequence>& sequences,
                  int numIterations = 100) {
    std::cout << "\n=== Running Multi-Output Benchmark ===\n";
    std::cout << "Number of sequences: " << sequences.size() << "\n";
    std::cout << "Number of iterations: " << numIterations << "\n\n";
    
    tagger.resetMetrics();
    
    auto overallStart = std::chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < numIterations; iter++) {
        for (const auto& seq : sequences) {
            tagger.predict(seq);
        }
        
        if ((iter + 1) % 10 == 0) {
            std::cout << "Completed " << (iter + 1) << "/" << numIterations << " iterations\n";
        }
    }
    
    auto overallEnd = std::chrono::high_resolution_clock::now();
    double totalTime = std::chrono::duration<double, std::milli>(overallEnd - overallStart).count();
    
    std::cout << "\n=== Benchmark Results ===\n";
    auto metrics = tagger.getMetrics();
    metrics.print();
    
    std::cout << "\nOverall Statistics:\n";
    std::cout << "  Total Predictions: " << (sequences.size() * numIterations) << "\n";
    std::cout << "  Total Time: " << std::fixed << std::setprecision(2) << totalTime << " ms\n";
    std::cout << "  Avg Time per Batch: " << (totalTime / numIterations) << " ms\n";
}

void runDemo(RadarTaggerMultiOutput& tagger) {
    std::cout << "\n=== Running Demo with Synthetic Data ===\n";
    
    // Create synthetic radar sequence
    RadarSequence testSeq;
    testSeq.trackId = 1;
    testSeq.sequenceLength = 20;
    
    for (int i = 0; i < 20; i++) {
        RadarDataPoint point;
        point.time = i * 0.1f;
        point.trackid = 1.0f;
        point.x = 10000.0f + i * 50.0f;  // Moving away (outgoing)
        point.y = 10000.0f;
        point.z = 2000.0f;                // Constant altitude (level)
        point.vx = 50.0f;                 // Constant velocity (linear, low_speed)
        point.vy = 0.0f;
        point.vz = 0.0f;
        point.ax = 0.0f;                  // No acceleration (light_maneuver)
        point.ay = 0.0f;
        point.az = 0.0f;
        point.speed = 50.0f;
        point.speed_2d = 50.0f;
        point.heading = 0.0f;
        point.range = 14000.0f + i * 5.0f;
        point.range_rate = 5.0f;          // Moving away
        point.curvature = 0.0f;           // Straight path
        point.accel_magnitude = 0.0f;
        point.vertical_rate = 0.0f;
        point.altitude_change = 0.0f;
        
        testSeq.points.push_back(point);
    }
    
    std::cout << "Running inference on synthetic data...\n";
    std::cout << "Expected tags: outgoing, level_flight, linear, light_maneuver, low_speed\n\n";
    
    auto result = tagger.predict(testSeq);
    
    if (result.success) {
        std::cout << "=== Prediction Result ===\n";
        std::cout << "Aggregated Label: " << result.aggregatedLabel << "\n";
        std::cout << "Inference Time: " << std::fixed << std::setprecision(3) 
                  << result.inferenceTimeMs << " ms\n\n";
        
        result.tags.print();
    } else {
        std::cerr << "Prediction failed: " << result.errorMessage << "\n";
    }
}

int main(int argc, char* argv[]) {
    std::cout << "==============================================\n";
    std::cout << "  Multi-Output Radar Trajectory Tagger\n";
    std::cout << "  Supports XGBoost, Random Forest, Neural Networks\n";
    std::cout << "==============================================\n\n";
    
    // Parse command line arguments
    std::string modelPath;
    std::string metadataPath;
    std::string testDataPath;
    std::string modelTypeStr = "nn";
    bool isBinary = false;
    bool loadGroundTruth = false;
    int numSamples = 10;
    int seqLength = 20;
    int numFeatures = 18;
    int numThreads = 4;
    bool benchmark = false;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--model" && i + 1 < argc) {
            modelPath = argv[++i];
        } else if (arg == "--metadata" && i + 1 < argc) {
            metadataPath = argv[++i];
        } else if (arg == "--model-type" && i + 1 < argc) {
            modelTypeStr = argv[++i];
        } else if (arg == "--test-data" && i + 1 < argc) {
            testDataPath = argv[++i];
        } else if (arg == "--test-binary") {
            isBinary = true;
        } else if (arg == "--load-gt") {
            loadGroundTruth = true;
        } else if (arg == "--samples" && i + 1 < argc) {
            numSamples = std::stoi(argv[++i]);
        } else if (arg == "--seq-length" && i + 1 < argc) {
            seqLength = std::stoi(argv[++i]);
        } else if (arg == "--features" && i + 1 < argc) {
            numFeatures = std::stoi(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            numThreads = std::stoi(argv[++i]);
        } else if (arg == "--benchmark") {
            benchmark = true;
        }
    }
    
    // Validate required arguments
    if (modelPath.empty() || metadataPath.empty()) {
        std::cerr << "Error: --model and --metadata are required\n\n";
        printUsage(argv[0]);
        return 1;
    }
    
    // Determine model type
    ModelType modelType = ModelType::NEURAL_NETWORK;
    if (modelTypeStr == "xgboost" || modelTypeStr == "xgb") {
        modelType = ModelType::XGBOOST;
    } else if (modelTypeStr == "rf" || modelTypeStr == "randomforest") {
        modelType = ModelType::RANDOM_FOREST;
    }
    
    // Create tagger
    RadarTaggerMultiOutput tagger(modelPath, metadataPath, modelType, numThreads);
    
    // Initialize
    if (!tagger.initialize()) {
        std::cerr << "Failed to initialize tagger\n";
        return 1;
    }
    
    // Load test data if provided
    if (!testDataPath.empty()) {
        std::vector<RadarSequence> sequences;
        std::vector<MultiOutputTags> groundTruths;
        
        if (isBinary) {
            sequences = RadarTaggerMultiOutput::loadFromBinary(testDataPath, numSamples, seqLength, numFeatures);
        } else {
            auto data = RadarTaggerMultiOutput::loadFromCSV(testDataPath, loadGroundTruth);
            sequences = data.first;
            groundTruths = data.second;
        }
        
        if (sequences.empty()) {
            std::cerr << "No test data loaded\n";
            return 1;
        }
        
        if (benchmark) {
            runBenchmark(tagger, sequences);
        } else {
            runEvaluation(tagger, sequences, groundTruths);
        }
    } else {
        // Run demo with synthetic data
        runDemo(tagger);
    }
    
    std::cout << "\n==============================================\n";
    std::cout << "  Multi-Output Inference Complete\n";
    std::cout << "==============================================\n";
    
    return 0;
}

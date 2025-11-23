/**
 * @file main.cpp
 * @brief Main application for radar trajectory real-time tagging
 */

#include "radar_tagger.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>

void printUsage(const char* progName) {
    std::cout << "Usage: " << progName << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --model PATH        Path to TFLite model file (required)\n";
    std::cout << "  --metadata PATH     Path to model metadata JSON (required)\n";
    std::cout << "  --test-data PATH    Path to test data (CSV or binary)\n";
    std::cout << "  --test-binary       Test data is in binary format\n";
    std::cout << "  --samples N         Number of samples in binary file (default: 10)\n";
    std::cout << "  --seq-length N      Sequence length for binary file (default: 20)\n";
    std::cout << "  --features N        Number of features for binary file (default: 18)\n";
    std::cout << "  --threads N         Number of threads for inference (default: 4)\n";
    std::cout << "  --benchmark         Run benchmark mode\n";
    std::cout << "  --help              Show this help message\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << progName << " --model model.tflite --metadata metadata.json\n";
    std::cout << "  " << progName << " --model model.tflite --metadata metadata.json --test-data data.csv\n";
    std::cout << "  " << progName << " --model model.tflite --metadata metadata.json --test-data data.bin --test-binary\n";
}

void runBenchmark(RadarTagger& tagger, const std::vector<RadarSequence>& sequences, int numIterations = 100) {
    std::cout << "\n=== Running Benchmark ===\n";
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

void evaluateModel(RadarTagger& tagger, const std::vector<RadarSequence>& sequences) {
    std::cout << "\n=== Evaluating Model ===\n";
    std::cout << "Number of test sequences: " << sequences.size() << "\n\n";
    
    tagger.resetMetrics();
    
    std::map<int, int> classCounts;
    std::map<int, std::vector<float>> classConfidences;
    
    for (size_t i = 0; i < sequences.size(); i++) {
        auto result = tagger.predict(sequences[i]);
        
        if (result.success) {
            classCounts[result.predictedClass]++;
            float maxProb = *std::max_element(result.classProbabilities.begin(),
                                             result.classProbabilities.end());
            classConfidences[result.predictedClass].push_back(maxProb);
            
            std::cout << "Sequence " << i << " (Track " << sequences[i].trackId << "): ";
            std::cout << result.className << " (Class " << result.predictedClass << ") ";
            std::cout << "Confidence: " << std::fixed << std::setprecision(3) << maxProb << " ";
            std::cout << "Time: " << std::setprecision(2) << result.inferenceTimeMs << " ms\n";
        } else {
            std::cerr << "Sequence " << i << " failed: " << result.errorMessage << "\n";
        }
    }
    
    std::cout << "\n=== Class Distribution ===\n";
    auto classNames = tagger.getClassNames();
    for (const auto& [classId, count] : classCounts) {
        std::string className = classId < classNames.size() ? classNames[classId] : "Unknown";
        float avgConfidence = 0.0f;
        if (!classConfidences[classId].empty()) {
            avgConfidence = std::accumulate(classConfidences[classId].begin(),
                                           classConfidences[classId].end(), 0.0f) / 
                           classConfidences[classId].size();
        }
        
        std::cout << "  Class " << classId << " (" << className << "): " << count;
        std::cout << " predictions, Avg Confidence: " << std::fixed << std::setprecision(3) 
                  << avgConfidence << "\n";
    }
    
    auto metrics = tagger.getMetrics();
    metrics.print();
}

int main(int argc, char* argv[]) {
    std::cout << "==============================================\n";
    std::cout << "  Radar Trajectory Real-Time Tagger\n";
    std::cout << "  Using TensorFlow Lite for C++ Deployment\n";
    std::cout << "==============================================\n\n";
    
    // Parse command line arguments
    std::string modelPath;
    std::string metadataPath;
    std::string testDataPath;
    bool isBinary = false;
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
        } else if (arg == "--test-data" && i + 1 < argc) {
            testDataPath = argv[++i];
        } else if (arg == "--test-binary") {
            isBinary = true;
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
    
    // Create tagger
    RadarTagger tagger(modelPath, metadataPath, numThreads);
    
    // Initialize
    if (!tagger.initialize()) {
        std::cerr << "Failed to initialize tagger\n";
        return 1;
    }
    
    // Load test data if provided
    if (!testDataPath.empty()) {
        std::vector<RadarSequence> sequences;
        
        if (isBinary) {
            sequences = RadarTagger::loadFromBinary(testDataPath, numSamples, seqLength, numFeatures);
        } else {
            sequences = RadarTagger::loadFromCSV(testDataPath);
        }
        
        if (sequences.empty()) {
            std::cerr << "No test data loaded\n";
            return 1;
        }
        
        if (benchmark) {
            runBenchmark(tagger, sequences);
        } else {
            evaluateModel(tagger, sequences);
        }
    } else {
        // Create synthetic test data
        std::cout << "\nNo test data provided. Creating synthetic data for demonstration...\n";
        
        RadarSequence testSeq;
        testSeq.trackId = 1;
        testSeq.sequenceLength = 20;
        
        // Create some dummy radar data
        for (int i = 0; i < 20; i++) {
            RadarDataPoint point;
            point.time = i * 0.1f;
            point.trackid = 1.0f;
            point.x = 10000.0f + i * 50.0f;
            point.y = 10000.0f;
            point.z = 2000.0f;
            point.vx = 50.0f;
            point.vy = 0.0f;
            point.vz = 0.0f;
            point.ax = 0.0f;
            point.ay = 0.0f;
            point.az = 0.0f;
            point.speed = 50.0f;
            point.speed_2d = 50.0f;
            point.heading = 0.0f;
            point.range = 14000.0f + i * 10.0f;
            point.range_rate = -10.0f;
            point.curvature = 0.0f;
            point.accel_magnitude = 0.0f;
            point.vertical_rate = 0.0f;
            point.altitude_change = 0.0f;
            
            testSeq.points.push_back(point);
        }
        
        std::cout << "\nRunning inference on synthetic data...\n";
        auto result = tagger.predict(testSeq);
        
        if (result.success) {
            std::cout << "\n=== Prediction Result ===\n";
            std::cout << "Predicted Class: " << result.predictedClass << "\n";
            std::cout << "Class Name: " << result.className << "\n";
            std::cout << "Inference Time: " << std::fixed << std::setprecision(3) 
                      << result.inferenceTimeMs << " ms\n";
            
            std::cout << "\nClass Probabilities:\n";
            auto classNames = tagger.getClassNames();
            for (size_t i = 0; i < result.classProbabilities.size(); i++) {
                std::string className = i < classNames.size() ? classNames[i] : "Unknown";
                std::cout << "  Class " << i << " (" << className << "): "
                          << std::fixed << std::setprecision(4) << result.classProbabilities[i] << "\n";
            }
        } else {
            std::cerr << "Prediction failed: " << result.errorMessage << "\n";
        }
    }
    
    std::cout << "\n==============================================\n";
    std::cout << "  Inference Complete\n";
    std::cout << "==============================================\n";
    
    return 0;
}

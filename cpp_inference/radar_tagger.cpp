/**
 * @file radar_tagger.cpp
 * @brief Implementation of Real-time radar trajectory tagger
 */

#include "radar_tagger.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iomanip>

// JSON parsing (simple implementation)
#include "json.hpp"
using json = nlohmann::json;

// Convert RadarDataPoint to feature vector
std::vector<float> RadarDataPoint::toFeatureVector() const {
    return {
        x, y, z, vx, vy, vz, ax, ay, az,
        speed, speed_2d, heading, range, range_rate,
        curvature, accel_magnitude, vertical_rate, altitude_change
    };
}

// Prepare sequence for model input
std::vector<float> RadarSequence::prepareModelInput(int targetLength) const {
    std::vector<float> input;
    
    int actualLength = std::min((int)points.size(), targetLength);
    int padding = targetLength - actualLength;
    
    // Add padding at the beginning if needed
    for (int i = 0; i < padding; i++) {
        // Pad with zeros
        for (int j = 0; j < 18; j++) {  // 18 features
            input.push_back(0.0f);
        }
    }
    
    // Add actual data points
    for (int i = 0; i < actualLength; i++) {
        auto features = points[i].toFeatureVector();
        input.insert(input.end(), features.begin(), features.end());
    }
    
    return input;
}

// Print performance metrics
void PerformanceMetrics::print() const {
    std::cout << "\n=== Performance Metrics ===\n";
    std::cout << "Total Inferences: " << totalInferences << "\n";
    std::cout << "Average Inference Time: " << avgInferenceTimeMs << " ms\n";
    std::cout << "Min Inference Time: " << minInferenceTimeMs << " ms\n";
    std::cout << "Max Inference Time: " << maxInferenceTimeMs << " ms\n";
    std::cout << "Total Time: " << totalTimeMs << " ms\n";
    std::cout << "Throughput: " << std::fixed << std::setprecision(2) 
              << throughput << " inferences/sec\n";
}

// Constructor
RadarTagger::RadarTagger(const std::string& modelPath,
                         const std::string& metadataPath,
                         int numThreads)
    : modelPath_(modelPath),
      metadataPath_(metadataPath),
      numThreads_(numThreads),
      numClasses_(0),
      sequenceLength_(20),
      numFeatures_(18) {
    
    // Initialize metrics
    resetMetrics();
}

// Destructor
RadarTagger::~RadarTagger() {
    // Cleanup handled by unique_ptr
}

// Initialize the model
bool RadarTagger::initialize() {
    std::cout << "Initializing Radar Tagger...\n";
    std::cout << "Model path: " << modelPath_ << "\n";
    std::cout << "Metadata path: " << metadataPath_ << "\n";
    
    // Load metadata
    if (!loadMetadata()) {
        std::cerr << "Failed to load metadata\n";
        return false;
    }
    
    // Load model
    model_ = tflite::FlatBufferModel::BuildFromFile(modelPath_.c_str());
    if (!model_) {
        std::cerr << "Failed to load model from: " << modelPath_ << "\n";
        return false;
    }
    std::cout << "Model loaded successfully\n";
    
    // Build interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model_, resolver);
    builder(&interpreter_);
    
    if (!interpreter_) {
        std::cerr << "Failed to create interpreter\n";
        return false;
    }
    
    // Set number of threads
    interpreter_->SetNumThreads(numThreads_);
    std::cout << "Using " << numThreads_ << " threads for inference\n";
    
    // Allocate tensors
    if (interpreter_->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors\n";
        return false;
    }
    
    std::cout << "Tensors allocated successfully\n";
    
    // Print model info
    printModelInfo();
    
    return true;
}

// Load metadata
bool RadarTagger::loadMetadata() {
    std::cout << "Loading metadata from: " << metadataPath_ << "\n";
    
    std::ifstream metadataFile(metadataPath_);
    if (!metadataFile.is_open()) {
        std::cerr << "Failed to open metadata file\n";
        return false;
    }
    
    json metadata;
    try {
        metadataFile >> metadata;
    } catch (const std::exception& e) {
        std::cerr << "Failed to parse JSON: " << e.what() << "\n";
        return false;
    }
    
    // Load class names
    if (metadata.contains("classes")) {
        classNames_ = metadata["classes"].get<std::vector<std::string>>();
        numClasses_ = classNames_.size();
        std::cout << "Loaded " << numClasses_ << " class names\n";
    }
    
    // Load scaler parameters
    if (metadata.contains("scaler_mean")) {
        scalerMean_ = metadata["scaler_mean"].get<std::vector<float>>();
    }
    if (metadata.contains("scaler_scale")) {
        scalerScale_ = metadata["scaler_scale"].get<std::vector<float>>();
    }
    
    // Load feature columns
    if (metadata.contains("feature_columns")) {
        featureColumns_ = metadata["feature_columns"].get<std::vector<std::string>>();
        numFeatures_ = featureColumns_.size();
    }
    
    // Load sequence length
    if (metadata.contains("sequence_length")) {
        sequenceLength_ = metadata["sequence_length"].get<int>();
    }
    
    std::cout << "Metadata loaded: " << numFeatures_ << " features, "
              << "sequence length " << sequenceLength_ << "\n";
    
    return true;
}

// Normalize input
std::vector<float> RadarTagger::normalizeInput(const std::vector<float>& input) {
    if (scalerMean_.empty() || scalerScale_.empty()) {
        return input;  // No normalization
    }
    
    std::vector<float> normalized(input.size());
    int featuresPerPoint = scalerMean_.size();
    
    for (size_t i = 0; i < input.size(); i++) {
        int featureIdx = i % featuresPerPoint;
        if (featureIdx < scalerMean_.size()) {
            normalized[i] = (input[i] - scalerMean_[featureIdx]) / scalerScale_[featureIdx];
        } else {
            normalized[i] = input[i];
        }
    }
    
    return normalized;
}

// Predict
PredictionResult RadarTagger::predict(const RadarSequence& sequence) {
    PredictionResult result;
    result.success = false;
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Prepare input
    auto inputData = sequence.prepareModelInput(sequenceLength_);
    
    // Normalize input
    auto normalizedInput = normalizeInput(inputData);
    
    // Get input tensor
    int inputIdx = interpreter_->inputs()[0];
    TfLiteTensor* inputTensor = interpreter_->tensor(inputIdx);
    
    // Check input size
    size_t expectedSize = inputTensor->bytes / sizeof(float);
    if (normalizedInput.size() != expectedSize) {
        result.errorMessage = "Input size mismatch. Expected: " + std::to_string(expectedSize) +
                            ", Got: " + std::to_string(normalizedInput.size());
        return result;
    }
    
    // Copy input data
    std::memcpy(inputTensor->data.f, normalizedInput.data(), inputTensor->bytes);
    
    // Run inference
    if (interpreter_->Invoke() != kTfLiteOk) {
        result.errorMessage = "Inference failed";
        return result;
    }
    
    // Get output
    int outputIdx = interpreter_->outputs()[0];
    TfLiteTensor* outputTensor = interpreter_->tensor(outputIdx);
    
    // Get output size
    int outputSize = outputTensor->dims->data[outputTensor->dims->size - 1];
    
    // Copy output probabilities
    result.classProbabilities.resize(outputSize);
    std::memcpy(result.classProbabilities.data(), outputTensor->data.f, 
                outputSize * sizeof(float));
    
    // Find predicted class (argmax)
    result.predictedClass = std::distance(
        result.classProbabilities.begin(),
        std::max_element(result.classProbabilities.begin(), result.classProbabilities.end())
    );
    
    // Get class name
    if (result.predictedClass < classNames_.size()) {
        result.className = classNames_[result.predictedClass];
    } else {
        result.className = "Class_" + std::to_string(result.predictedClass);
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    result.inferenceTimeMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    
    // Update metrics
    updateMetrics(result.inferenceTimeMs);
    
    result.success = true;
    return result;
}

// Batch prediction
std::vector<PredictionResult> RadarTagger::predictBatch(const std::vector<RadarSequence>& sequences) {
    std::vector<PredictionResult> results;
    results.reserve(sequences.size());
    
    for (const auto& sequence : sequences) {
        results.push_back(predict(sequence));
    }
    
    return results;
}

// Update metrics
void RadarTagger::updateMetrics(double inferenceTime) {
    inferenceTimes_.push_back(inferenceTime);
    metrics_.totalInferences++;
    metrics_.totalTimeMs += inferenceTime;
    
    if (metrics_.totalInferences == 1) {
        metrics_.minInferenceTimeMs = inferenceTime;
        metrics_.maxInferenceTimeMs = inferenceTime;
    } else {
        metrics_.minInferenceTimeMs = std::min(metrics_.minInferenceTimeMs, inferenceTime);
        metrics_.maxInferenceTimeMs = std::max(metrics_.maxInferenceTimeMs, inferenceTime);
    }
    
    metrics_.avgInferenceTimeMs = metrics_.totalTimeMs / metrics_.totalInferences;
    metrics_.throughput = 1000.0 / metrics_.avgInferenceTimeMs;  // inferences per second
}

// Reset metrics
void RadarTagger::resetMetrics() {
    metrics_ = PerformanceMetrics();
    metrics_.minInferenceTimeMs = 1e9;
    metrics_.maxInferenceTimeMs = 0;
    inferenceTimes_.clear();
}

// Print model info
void RadarTagger::printModelInfo() const {
    std::cout << "\n=== Model Information ===\n";
    std::cout << "Number of classes: " << numClasses_ << "\n";
    std::cout << "Sequence length: " << sequenceLength_ << "\n";
    std::cout << "Number of features: " << numFeatures_ << "\n";
    std::cout << "Number of threads: " << numThreads_ << "\n";
    
    if (!classNames_.empty()) {
        std::cout << "Class names:\n";
        for (size_t i = 0; i < classNames_.size(); i++) {
            std::cout << "  " << i << ": " << classNames_[i] << "\n";
        }
    }
    
    // Print tensor info
    std::cout << "\nInput tensors:\n";
    for (int i : interpreter_->inputs()) {
        TfLiteTensor* tensor = interpreter_->tensor(i);
        std::cout << "  Tensor " << i << ": " << tensor->name << " ";
        std::cout << "Shape: [";
        for (int d = 0; d < tensor->dims->size; d++) {
            std::cout << tensor->dims->data[d];
            if (d < tensor->dims->size - 1) std::cout << ", ";
        }
        std::cout << "]\n";
    }
    
    std::cout << "\nOutput tensors:\n";
    for (int i : interpreter_->outputs()) {
        TfLiteTensor* tensor = interpreter_->tensor(i);
        std::cout << "  Tensor " << i << ": " << tensor->name << " ";
        std::cout << "Shape: [";
        for (int d = 0; d < tensor->dims->size; d++) {
            std::cout << tensor->dims->data[d];
            if (d < tensor->dims->size - 1) std::cout << ", ";
        }
        std::cout << "]\n";
    }
}

// Get metrics
PerformanceMetrics RadarTagger::getMetrics() const {
    return metrics_;
}

// Load from CSV
std::vector<RadarSequence> RadarTagger::loadFromCSV(const std::string& csvPath) {
    std::vector<RadarSequence> sequences;
    
    std::ifstream file(csvPath);
    if (!file.is_open()) {
        std::cerr << "Failed to open CSV file: " << csvPath << "\n";
        return sequences;
    }
    
    std::string line;
    std::getline(file, line);  // Skip header
    
    std::map<int, std::vector<RadarDataPoint>> trackMap;
    
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        RadarDataPoint point;
        
        // Parse CSV line (assuming the format from the data file)
        char comma;
        ss >> point.time >> comma
           >> point.trackid >> comma
           >> point.x >> comma >> point.y >> comma >> point.z >> comma
           >> point.vx >> comma >> point.vy >> comma >> point.vz >> comma
           >> point.ax >> comma >> point.ay >> comma >> point.az >> comma
           >> point.speed >> comma >> point.speed_2d >> comma
           >> point.heading >> comma >> point.range >> comma >> point.range_rate >> comma
           >> point.curvature >> comma >> point.accel_magnitude >> comma
           >> point.vertical_rate >> comma >> point.altitude_change;
        
        int trackId = static_cast<int>(point.trackid);
        trackMap[trackId].push_back(point);
    }
    
    // Convert to sequences
    for (const auto& [trackId, points] : trackMap) {
        RadarSequence seq;
        seq.trackId = trackId;
        seq.points = points;
        seq.sequenceLength = points.size();
        sequences.push_back(seq);
    }
    
    std::cout << "Loaded " << sequences.size() << " sequences from CSV\n";
    return sequences;
}

// Load from binary
std::vector<RadarSequence> RadarTagger::loadFromBinary(const std::string& binPath,
                                                       int nSamples,
                                                       int seqLength,
                                                       int nFeatures) {
    std::vector<RadarSequence> sequences;
    
    std::ifstream file(binPath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open binary file: " << binPath << "\n";
        return sequences;
    }
    
    // Calculate total size
    size_t totalFloats = nSamples * seqLength * nFeatures;
    std::vector<float> data(totalFloats);
    
    // Read all data
    file.read(reinterpret_cast<char*>(data.data()), totalFloats * sizeof(float));
    
    if (!file) {
        std::cerr << "Failed to read complete binary file\n";
        return sequences;
    }
    
    // Convert to sequences
    for (int i = 0; i < nSamples; i++) {
        RadarSequence seq;
        seq.trackId = i;
        seq.sequenceLength = seqLength;
        
        for (int j = 0; j < seqLength; j++) {
            RadarDataPoint point;
            int baseIdx = (i * seqLength + j) * nFeatures;
            
            // Assign features (assuming they're in order)
            point.x = data[baseIdx + 0];
            point.y = data[baseIdx + 1];
            point.z = data[baseIdx + 2];
            point.vx = data[baseIdx + 3];
            point.vy = data[baseIdx + 4];
            point.vz = data[baseIdx + 5];
            point.ax = data[baseIdx + 6];
            point.ay = data[baseIdx + 7];
            point.az = data[baseIdx + 8];
            point.speed = data[baseIdx + 9];
            point.speed_2d = data[baseIdx + 10];
            point.heading = data[baseIdx + 11];
            point.range = data[baseIdx + 12];
            point.range_rate = data[baseIdx + 13];
            point.curvature = data[baseIdx + 14];
            point.accel_magnitude = data[baseIdx + 15];
            point.vertical_rate = data[baseIdx + 16];
            point.altitude_change = data[baseIdx + 17];
            
            seq.points.push_back(point);
        }
        
        sequences.push_back(seq);
    }
    
    std::cout << "Loaded " << sequences.size() << " sequences from binary file\n";
    return sequences;
}

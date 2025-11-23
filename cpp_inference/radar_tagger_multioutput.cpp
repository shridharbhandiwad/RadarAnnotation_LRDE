/**
 * @file radar_tagger_multioutput.cpp
 * @brief Implementation of multi-output radar trajectory tagger
 */

#include "radar_tagger_multioutput.h"
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

// JSON parsing
#include "json.hpp"
using json = nlohmann::json;

// Tag names in order
static const std::vector<std::string> TAG_NAMES = {
    "incoming", "outgoing",
    "fixed_range_ascending", "fixed_range_descending", "level_flight",
    "linear", "curved",
    "light_maneuver", "high_maneuver",
    "low_speed", "high_speed"
};

// === MultiOutputTags Implementation ===

std::string MultiOutputTags::toAggregatedLabel() const {
    std::vector<std::string> activeTags = getActiveTags();
    if (activeTags.empty()) {
        return "unknown";
    }
    
    std::string label;
    for (size_t i = 0; i < activeTags.size(); i++) {
        label += activeTags[i];
        if (i < activeTags.size() - 1) {
            label += ",";
        }
    }
    return label;
}

std::vector<std::string> MultiOutputTags::getActiveTags() const {
    std::vector<std::string> active;
    
    if (incoming) active.push_back("incoming");
    if (outgoing) active.push_back("outgoing");
    if (fixed_range_ascending) active.push_back("fixed_range_ascending");
    if (fixed_range_descending) active.push_back("fixed_range_descending");
    if (level_flight) active.push_back("level_flight");
    if (linear) active.push_back("linear");
    if (curved) active.push_back("curved");
    if (light_maneuver) active.push_back("light_maneuver");
    if (high_maneuver) active.push_back("high_maneuver");
    if (low_speed) active.push_back("low_speed");
    if (high_speed) active.push_back("high_speed");
    
    return active;
}

void MultiOutputTags::print() const {
    std::cout << "Tags: " << toAggregatedLabel() << "\n";
    std::cout << "Confidences:\n";
    for (const auto& [tag, conf] : confidences) {
        std::cout << "  " << tag << ": " << std::fixed << std::setprecision(3) << conf << "\n";
    }
}

// === RadarDataPoint Implementation ===

std::vector<float> RadarDataPoint::toFeatureVector() const {
    return {
        x, y, z, vx, vy, vz, ax, ay, az,
        speed, speed_2d, heading, range, range_rate,
        curvature, accel_magnitude, vertical_rate, altitude_change
    };
}

// === RadarSequence Implementation ===

std::vector<float> RadarSequence::prepareModelInput(int targetLength) const {
    std::vector<float> input;
    
    int actualLength = std::min((int)points.size(), targetLength);
    int padding = targetLength - actualLength;
    
    // Add padding at the beginning if needed
    for (int i = 0; i < padding; i++) {
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

std::vector<float> RadarSequence::computeAggregatedFeatures() const {
    // Compute statistical features across the sequence for non-sequential models
    if (points.empty()) {
        return std::vector<float>(18, 0.0f);
    }
    
    // Use the most recent point's features (could also do mean/max/min aggregation)
    return points.back().toFeatureVector();
}

// === MultiOutputMetrics Implementation ===

void MultiOutputMetrics::print() const {
    std::cout << "\n=== Multi-Output Performance Metrics ===\n";
    std::cout << "Total Inferences: " << totalInferences << "\n";
    std::cout << "Average Inference Time: " << std::fixed << std::setprecision(3) 
              << avgInferenceTimeMs << " ms\n";
    std::cout << "Min Inference Time: " << minInferenceTimeMs << " ms\n";
    std::cout << "Max Inference Time: " << maxInferenceTimeMs << " ms\n";
    std::cout << "Total Time: " << totalTimeMs << " ms\n";
    std::cout << "Throughput: " << std::setprecision(2) << throughput << " inferences/sec\n";
    
    if (!tagAccuracy.empty()) {
        std::cout << "\n=== Per-Tag Accuracy ===\n";
        for (const auto& [tag, acc] : tagAccuracy) {
            std::cout << "  " << std::setw(25) << std::left << tag << ": " 
                      << std::fixed << std::setprecision(1) << (acc * 100.0f) << "%\n";
        }
        std::cout << "\nOverall Accuracy: " << std::fixed << std::setprecision(1) 
                  << (overallAccuracy * 100.0f) << "%\n";
        std::cout << "Average F1 Score: " << std::fixed << std::setprecision(3) 
                  << averageF1Score << "\n";
    }
}

void MultiOutputMetrics::computeMetrics() {
    // Compute per-tag accuracy and F1 scores
    float totalF1 = 0.0f;
    int tagCount = 0;
    
    for (const auto& [tag, tp] : tagTruePositives) {
        int fp = tagFalsePositives[tag];
        int fn = tagFalseNegatives[tag];
        int tn = tagTrueNegatives[tag];
        
        int total = tp + fp + tn + fn;
        if (total > 0) {
            tagAccuracy[tag] = (float)(tp + tn) / total;
        }
        
        // Compute F1 score
        float precision = (tp + fp > 0) ? (float)tp / (tp + fp) : 0.0f;
        float recall = (tp + fn > 0) ? (float)tp / (tp + fn) : 0.0f;
        float f1 = (precision + recall > 0) ? 2.0f * precision * recall / (precision + recall) : 0.0f;
        
        totalF1 += f1;
        tagCount++;
    }
    
    if (tagCount > 0) {
        averageF1Score = totalF1 / tagCount;
        
        // Compute overall accuracy
        int totalCorrect = 0;
        int totalPredictions = 0;
        for (const auto& [tag, tp] : tagTruePositives) {
            totalCorrect += tp + tagTrueNegatives[tag];
            totalPredictions += tp + tagFalsePositives[tag] + tagTrueNegatives[tag] + tagFalseNegatives[tag];
        }
        overallAccuracy = totalPredictions > 0 ? (float)totalCorrect / totalPredictions : 0.0f;
    }
}

// === RadarTaggerMultiOutput Implementation ===

RadarTaggerMultiOutput::RadarTaggerMultiOutput(const std::string& modelPath,
                                               const std::string& metadataPath,
                                               ModelType modelType,
                                               int numThreads)
    : modelPath_(modelPath),
      metadataPath_(metadataPath),
      modelType_(modelType),
      numThreads_(numThreads),
      numTags_(11),
      sequenceLength_(20),
      numFeatures_(18),
      isSequenceModel_(modelType == ModelType::NEURAL_NETWORK) {
    
    tagNames_ = TAG_NAMES;
    resetMetrics();
}

RadarTaggerMultiOutput::~RadarTaggerMultiOutput() {
    // Cleanup handled by unique_ptr
}

bool RadarTaggerMultiOutput::initialize() {
    std::cout << "Initializing Multi-Output Radar Tagger...\n";
    std::cout << "Model path: " << modelPath_ << "\n";
    std::cout << "Metadata path: " << metadataPath_ << "\n";
    std::cout << "Model type: ";
    
    switch (modelType_) {
        case ModelType::NEURAL_NETWORK:
            std::cout << "Neural Network (TFLite)\n";
            break;
        case ModelType::XGBOOST:
            std::cout << "XGBoost\n";
            break;
        case ModelType::RANDOM_FOREST:
            std::cout << "Random Forest\n";
            break;
    }
    
    // Load metadata
    if (!loadMetadata()) {
        std::cerr << "Failed to load metadata\n";
        return false;
    }
    
    // Initialize model based on type
    if (modelType_ == ModelType::NEURAL_NETWORK) {
        // Load TFLite model
        model_ = tflite::FlatBufferModel::BuildFromFile(modelPath_.c_str());
        if (!model_) {
            std::cerr << "Failed to load TFLite model from: " << modelPath_ << "\n";
            return false;
        }
        std::cout << "TFLite model loaded successfully\n";
        
        // Build interpreter
        tflite::ops::builtin::BuiltinOpResolver resolver;
        tflite::InterpreterBuilder builder(*model_, resolver);
        builder(&interpreter_);
        
        if (!interpreter_) {
            std::cerr << "Failed to create interpreter\n";
            return false;
        }
        
        interpreter_->SetNumThreads(numThreads_);
        
        if (interpreter_->AllocateTensors() != kTfLiteOk) {
            std::cerr << "Failed to allocate tensors\n";
            return false;
        }
        
        std::cout << "TFLite interpreter initialized\n";
    } else {
        // For XGBoost/Random Forest, we would load the model here
        // This requires a C++ XGBoost library or custom implementation
        std::cout << "Note: XGBoost/RandomForest support requires additional libraries\n";
        std::cout << "Consider exporting to ONNX or TFLite for full C++ support\n";
        return false;  // Not fully implemented yet
    }
    
    printModelInfo();
    return true;
}

bool RadarTaggerMultiOutput::loadMetadata() {
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
    
    // Check if it's a multi-output model
    if (metadata.contains("metrics")) {
        auto metrics = metadata["metrics"];
        if (metrics.contains("multi_output")) {
            bool isMultiOutput = metrics["multi_output"].get<bool>();
            if (!isMultiOutput) {
                std::cerr << "Warning: Model metadata indicates single-output model\n";
            }
        }
    }
    
    std::cout << "Metadata loaded: " << numFeatures_ << " features, "
              << numTags_ << " tags, sequence length " << sequenceLength_ << "\n";
    
    return true;
}

std::vector<float> RadarTaggerMultiOutput::normalizeInput(const std::vector<float>& input) {
    if (scalerMean_.empty() || scalerScale_.empty()) {
        return input;
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

MultiOutputTags RadarTaggerMultiOutput::parseOutputTags(const std::vector<float>& outputs) {
    MultiOutputTags tags = {};
    
    // Assuming outputs are in the order of TAG_NAMES
    // Each output is a probability (0.0 to 1.0)
    float threshold = 0.5f;
    
    if (outputs.size() >= 11) {
        tags.incoming = outputs[0] > threshold;
        tags.outgoing = outputs[1] > threshold;
        tags.fixed_range_ascending = outputs[2] > threshold;
        tags.fixed_range_descending = outputs[3] > threshold;
        tags.level_flight = outputs[4] > threshold;
        tags.linear = outputs[5] > threshold;
        tags.curved = outputs[6] > threshold;
        tags.light_maneuver = outputs[7] > threshold;
        tags.high_maneuver = outputs[8] > threshold;
        tags.low_speed = outputs[9] > threshold;
        tags.high_speed = outputs[10] > threshold;
        
        // Store confidences
        for (size_t i = 0; i < std::min(outputs.size(), TAG_NAMES.size()); i++) {
            tags.confidences[TAG_NAMES[i]] = outputs[i];
        }
    }
    
    return tags;
}

MultiOutputResult RadarTaggerMultiOutput::predict(const RadarSequence& sequence,
                                                  const MultiOutputTags* groundTruth) {
    switch (modelType_) {
        case ModelType::NEURAL_NETWORK:
            return predictNeuralNetwork(sequence);
        case ModelType::XGBOOST:
            return predictXGBoost(sequence);
        case ModelType::RANDOM_FOREST:
            return predictRandomForest(sequence);
        default:
            MultiOutputResult result;
            result.success = false;
            result.errorMessage = "Unknown model type";
            return result;
    }
}

MultiOutputResult RadarTaggerMultiOutput::predictNeuralNetwork(const RadarSequence& sequence) {
    MultiOutputResult result;
    result.success = false;
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Prepare input
    auto inputData = sequence.prepareModelInput(sequenceLength_);
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
    
    // Get output (multi-output: 11 binary predictions)
    int outputIdx = interpreter_->outputs()[0];
    TfLiteTensor* outputTensor = interpreter_->tensor(outputIdx);
    
    // Extract outputs
    std::vector<float> outputs;
    int outputSize = 1;
    for (int i = 0; i < outputTensor->dims->size; i++) {
        outputSize *= outputTensor->dims->data[i];
    }
    
    outputs.resize(outputSize);
    std::memcpy(outputs.data(), outputTensor->data.f, outputSize * sizeof(float));
    
    // Parse tags
    result.tags = parseOutputTags(outputs);
    result.aggregatedLabel = result.tags.toAggregatedLabel();
    
    auto endTime = std::chrono::high_resolution_clock::now();
    result.inferenceTimeMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    
    updateMetrics(result.inferenceTimeMs, result);
    
    result.success = true;
    return result;
}

MultiOutputResult RadarTaggerMultiOutput::predictXGBoost(const RadarSequence& sequence) {
    MultiOutputResult result;
    result.success = false;
    result.errorMessage = "XGBoost prediction not implemented (requires XGBoost C++ library)";
    return result;
}

MultiOutputResult RadarTaggerMultiOutput::predictRandomForest(const RadarSequence& sequence) {
    MultiOutputResult result;
    result.success = false;
    result.errorMessage = "Random Forest prediction not implemented (requires RF C++ library)";
    return result;
}

std::vector<MultiOutputResult> RadarTaggerMultiOutput::predictBatch(
    const std::vector<RadarSequence>& sequences,
    const std::vector<MultiOutputTags>* groundTruths) {
    
    std::vector<MultiOutputResult> results;
    results.reserve(sequences.size());
    
    for (size_t i = 0; i < sequences.size(); i++) {
        const MultiOutputTags* gt = (groundTruths && i < groundTruths->size()) ? 
                                    &(*groundTruths)[i] : nullptr;
        results.push_back(predict(sequences[i], gt));
    }
    
    return results;
}

void RadarTaggerMultiOutput::updateMetrics(double inferenceTime, const MultiOutputResult& result) {
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
    metrics_.throughput = 1000.0 / metrics_.avgInferenceTimeMs;
    
    // Update tag-level metrics if ground truth available
    if (!result.groundTruth.empty()) {
        auto activeTags = result.tags.getActiveTags();
        std::set<std::string> predictedSet(activeTags.begin(), activeTags.end());
        
        for (const auto& tagName : tagNames_) {
            bool predicted = predictedSet.count(tagName) > 0;
            bool actual = result.groundTruth.count(tagName) > 0 && result.groundTruth.at(tagName);
            
            if (predicted && actual) {
                metrics_.tagTruePositives[tagName]++;
            } else if (predicted && !actual) {
                metrics_.tagFalsePositives[tagName]++;
            } else if (!predicted && actual) {
                metrics_.tagFalseNegatives[tagName]++;
            } else {
                metrics_.tagTrueNegatives[tagName]++;
            }
        }
    }
}

void RadarTaggerMultiOutput::resetMetrics() {
    metrics_ = MultiOutputMetrics();
    metrics_.minInferenceTimeMs = 1e9;
    metrics_.maxInferenceTimeMs = 0;
    inferenceTimes_.clear();
}

void RadarTaggerMultiOutput::printModelInfo() const {
    std::cout << "\n=== Multi-Output Model Information ===\n";
    std::cout << "Number of tags: " << numTags_ << "\n";
    std::cout << "Sequence length: " << sequenceLength_ << "\n";
    std::cout << "Number of features: " << numFeatures_ << "\n";
    std::cout << "Number of threads: " << numThreads_ << "\n";
    
    std::cout << "\nOutput Tags:\n";
    for (size_t i = 0; i < tagNames_.size(); i++) {
        std::cout << "  " << i << ": " << tagNames_[i] << "\n";
    }
    
    if (modelType_ == ModelType::NEURAL_NETWORK && interpreter_) {
        std::cout << "\nTensor Information:\n";
        std::cout << "Input tensors:\n";
        for (int i : interpreter_->inputs()) {
            TfLiteTensor* tensor = interpreter_->tensor(i);
            std::cout << "  " << tensor->name << " Shape: [";
            for (int d = 0; d < tensor->dims->size; d++) {
                std::cout << tensor->dims->data[d];
                if (d < tensor->dims->size - 1) std::cout << ", ";
            }
            std::cout << "]\n";
        }
        
        std::cout << "\nOutput tensors:\n";
        for (int i : interpreter_->outputs()) {
            TfLiteTensor* tensor = interpreter_->tensor(i);
            std::cout << "  " << tensor->name << " Shape: [";
            for (int d = 0; d < tensor->dims->size; d++) {
                std::cout << tensor->dims->data[d];
                if (d < tensor->dims->size - 1) std::cout << ", ";
            }
            std::cout << "]\n";
        }
    }
}

MultiOutputMetrics RadarTaggerMultiOutput::getMetrics() const {
    MultiOutputMetrics m = metrics_;
    m.computeMetrics();
    return m;
}

// Load from CSV (implementation)
std::pair<std::vector<RadarSequence>, std::vector<MultiOutputTags>>
RadarTaggerMultiOutput::loadFromCSV(const std::string& csvPath, bool loadGroundTruth) {
    std::vector<RadarSequence> sequences;
    std::vector<MultiOutputTags> groundTruths;
    
    std::ifstream file(csvPath);
    if (!file.is_open()) {
        std::cerr << "Failed to open CSV file: " << csvPath << "\n";
        return {sequences, groundTruths};
    }
    
    std::string line;
    std::getline(file, line);  // Read header
    
    std::map<int, std::vector<RadarDataPoint>> trackMap;
    std::map<int, MultiOutputTags> tagMap;
    
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<std::string> fields;
        std::string field;
        
        while (std::getline(ss, field, ',')) {
            fields.push_back(field);
        }
        
        if (fields.size() < 19) continue;
        
        RadarDataPoint point;
        point.time = std::stof(fields[0]);
        point.trackid = std::stof(fields[1]);
        point.x = std::stof(fields[2]);
        point.y = std::stof(fields[3]);
        point.z = std::stof(fields[4]);
        point.vx = std::stof(fields[5]);
        point.vy = std::stof(fields[6]);
        point.vz = std::stof(fields[7]);
        point.ax = std::stof(fields[8]);
        point.ay = std::stof(fields[9]);
        point.az = std::stof(fields[10]);
        point.speed = std::stof(fields[11]);
        point.speed_2d = std::stof(fields[12]);
        point.heading = std::stof(fields[13]);
        point.range = std::stof(fields[14]);
        point.range_rate = std::stof(fields[15]);
        point.curvature = std::stof(fields[16]);
        point.accel_magnitude = std::stof(fields[17]);
        point.vertical_rate = std::stof(fields[18]);
        point.altitude_change = fields.size() > 19 ? std::stof(fields[19]) : 0.0f;
        
        int trackId = static_cast<int>(point.trackid);
        trackMap[trackId].push_back(point);
        
        // Load ground truth tags if available (columns 20-30)
        if (loadGroundTruth && fields.size() >= 31) {
            MultiOutputTags tags = {};
            tags.incoming = std::stoi(fields[20]) > 0;
            tags.outgoing = std::stoi(fields[21]) > 0;
            tags.fixed_range_ascending = std::stoi(fields[22]) > 0;
            tags.fixed_range_descending = std::stoi(fields[23]) > 0;
            tags.level_flight = std::stoi(fields[24]) > 0;
            tags.linear = std::stoi(fields[25]) > 0;
            tags.curved = std::stoi(fields[26]) > 0;
            tags.light_maneuver = std::stoi(fields[27]) > 0;
            tags.high_maneuver = std::stoi(fields[28]) > 0;
            tags.low_speed = std::stoi(fields[29]) > 0;
            tags.high_speed = std::stoi(fields[30]) > 0;
            tagMap[trackId] = tags;
        }
    }
    
    // Convert to sequences
    for (const auto& [trackId, points] : trackMap) {
        RadarSequence seq;
        seq.trackId = trackId;
        seq.points = points;
        seq.sequenceLength = points.size();
        sequences.push_back(seq);
        
        if (loadGroundTruth && tagMap.count(trackId)) {
            groundTruths.push_back(tagMap[trackId]);
        }
    }
    
    std::cout << "Loaded " << sequences.size() << " sequences from CSV\n";
    if (loadGroundTruth) {
        std::cout << "Loaded " << groundTruths.size() << " ground truth labels\n";
    }
    
    return {sequences, groundTruths};
}

// Load from binary (same as before)
std::vector<RadarSequence> RadarTaggerMultiOutput::loadFromBinary(
    const std::string& binPath,
    int nSamples,
    int seqLength,
    int nFeatures) {
    
    std::vector<RadarSequence> sequences;
    
    std::ifstream file(binPath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open binary file: " << binPath << "\n";
        return sequences;
    }
    
    size_t totalFloats = nSamples * seqLength * nFeatures;
    std::vector<float> data(totalFloats);
    
    file.read(reinterpret_cast<char*>(data.data()), totalFloats * sizeof(float));
    
    if (!file) {
        std::cerr << "Failed to read complete binary file\n";
        return sequences;
    }
    
    for (int i = 0; i < nSamples; i++) {
        RadarSequence seq;
        seq.trackId = i;
        seq.sequenceLength = seqLength;
        
        for (int j = 0; j < seqLength; j++) {
            RadarDataPoint point;
            int baseIdx = (i * seqLength + j) * nFeatures;
            
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

/**
 * @file radar_tagger_multioutput.h
 * @brief Real-time radar trajectory tagger with multi-output predictions
 * Supports XGBoost, Random Forest, and Neural Network models
 */

#ifndef RADAR_TAGGER_MULTIOUTPUT_H
#define RADAR_TAGGER_MULTIOUTPUT_H

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <set>

// Forward declarations for TensorFlow Lite
namespace tflite {
    class FlatBufferModel;
    class Interpreter;
    class InterpreterBuilder;
}

/**
 * @brief Multi-output tags for radar trajectory classification
 */
struct MultiOutputTags {
    // Direction tags (mutually exclusive)
    bool incoming;
    bool outgoing;
    
    // Vertical motion tags
    bool fixed_range_ascending;
    bool fixed_range_descending;
    bool level_flight;
    
    // Path shape tags
    bool linear;
    bool curved;
    
    // Maneuver intensity tags
    bool light_maneuver;
    bool high_maneuver;
    
    // Speed tags
    bool low_speed;
    bool high_speed;
    
    // Confidence scores for each tag (0.0 to 1.0)
    std::map<std::string, float> confidences;
    
    // Create aggregated label from active tags
    std::string toAggregatedLabel() const;
    
    // Get all active tags
    std::vector<std::string> getActiveTags() const;
    
    // Print tags
    void print() const;
};

/**
 * @brief Radar data point structure
 */
struct RadarDataPoint {
    float time;
    float trackid;
    float x, y, z;
    float vx, vy, vz;
    float ax, ay, az;
    float speed, speed_2d;
    float heading, range, range_rate;
    float curvature, accel_magnitude;
    float vertical_rate, altitude_change;
    
    // Convert to feature vector for model input
    std::vector<float> toFeatureVector() const;
};

/**
 * @brief Radar trajectory sequence for classification
 */
struct RadarSequence {
    std::vector<RadarDataPoint> points;
    int trackId;
    int sequenceLength;
    
    // Prepare sequence for model input (padding/truncation)
    std::vector<float> prepareModelInput(int targetLength = 20) const;
    
    // Compute aggregated features for non-sequential models
    std::vector<float> computeAggregatedFeatures() const;
};

/**
 * @brief Multi-output prediction result
 */
struct MultiOutputResult {
    MultiOutputTags tags;
    std::string aggregatedLabel;
    double inferenceTimeMs;
    bool success;
    std::string errorMessage;
    
    // Tag prediction accuracy (if ground truth available)
    std::map<std::string, bool> groundTruth;
    std::map<std::string, bool> correctPredictions;
    float overallAccuracy;
};

/**
 * @brief Model type enumeration
 */
enum class ModelType {
    NEURAL_NETWORK,  // LSTM, Transformer (TFLite)
    XGBOOST,         // XGBoost (requires separate handling)
    RANDOM_FOREST    // Random Forest (requires separate handling)
};

/**
 * @brief Performance metrics for multi-output models
 */
struct MultiOutputMetrics {
    int totalInferences;
    double avgInferenceTimeMs;
    double minInferenceTimeMs;
    double maxInferenceTimeMs;
    double totalTimeMs;
    double throughput;
    
    // Per-tag accuracy
    std::map<std::string, float> tagAccuracy;
    std::map<std::string, int> tagTruePositives;
    std::map<std::string, int> tagFalsePositives;
    std::map<std::string, int> tagTrueNegatives;
    std::map<std::string, int> tagFalseNegatives;
    
    // Overall metrics
    float overallAccuracy;
    float averageF1Score;
    
    void print() const;
    void computeMetrics();
};

/**
 * @brief Real-time Radar Tagger with Multi-Output Support
 */
class RadarTaggerMultiOutput {
public:
    /**
     * @brief Construct a new Radar Tagger
     * @param modelPath Path to model file (.tflite for NN, .pkl for XGBoost/RF, or .json for XGBoost)
     * @param metadataPath Path to model metadata JSON file
     * @param modelType Type of model (NEURAL_NETWORK, XGBOOST, RANDOM_FOREST)
     * @param numThreads Number of threads for inference (default: 4)
     */
    RadarTaggerMultiOutput(const std::string& modelPath,
                          const std::string& metadataPath,
                          ModelType modelType = ModelType::NEURAL_NETWORK,
                          int numThreads = 4);
    
    /**
     * @brief Destructor
     */
    ~RadarTaggerMultiOutput();
    
    /**
     * @brief Initialize the model
     * @return true if successful, false otherwise
     */
    bool initialize();
    
    /**
     * @brief Predict multi-output tags for a radar sequence
     * @param sequence Input radar sequence
     * @param groundTruth Optional ground truth tags for evaluation
     * @return Multi-output prediction result
     */
    MultiOutputResult predict(const RadarSequence& sequence,
                             const MultiOutputTags* groundTruth = nullptr);
    
    /**
     * @brief Batch prediction for multiple sequences
     * @param sequences Vector of radar sequences
     * @param groundTruths Optional ground truth tags for evaluation
     * @return Vector of multi-output prediction results
     */
    std::vector<MultiOutputResult> predictBatch(
        const std::vector<RadarSequence>& sequences,
        const std::vector<MultiOutputTags>* groundTruths = nullptr);
    
    /**
     * @brief Load test data from CSV file
     * @param csvPath Path to CSV file
     * @param loadGroundTruth Whether to load ground truth labels
     * @return Pair of sequences and optional ground truth tags
     */
    static std::pair<std::vector<RadarSequence>, std::vector<MultiOutputTags>>
        loadFromCSV(const std::string& csvPath, bool loadGroundTruth = false);
    
    /**
     * @brief Load test data from binary file
     * @param binPath Path to binary file
     * @param nSamples Number of samples
     * @param seqLength Sequence length
     * @param nFeatures Number of features
     * @return Vector of radar sequences
     */
    static std::vector<RadarSequence> loadFromBinary(const std::string& binPath,
                                                     int nSamples,
                                                     int seqLength,
                                                     int nFeatures);
    
    /**
     * @brief Get model information
     */
    void printModelInfo() const;
    
    /**
     * @brief Get performance metrics
     */
    MultiOutputMetrics getMetrics() const;
    
    /**
     * @brief Reset performance metrics
     */
    void resetMetrics();
    
    /**
     * @brief Get tag names
     */
    std::vector<std::string> getTagNames() const { return tagNames_; }
    
    /**
     * @brief Get model type
     */
    ModelType getModelType() const { return modelType_; }

private:
    std::string modelPath_;
    std::string metadataPath_;
    ModelType modelType_;
    int numThreads_;
    
    // TensorFlow Lite components (for neural networks)
    std::unique_ptr<tflite::FlatBufferModel> model_;
    std::unique_ptr<tflite::Interpreter> interpreter_;
    
    // Model metadata
    std::vector<std::string> tagNames_;
    std::vector<float> scalerMean_;
    std::vector<float> scalerScale_;
    std::vector<std::string> featureColumns_;
    int numTags_;
    int sequenceLength_;
    int numFeatures_;
    bool isSequenceModel_;  // True for LSTM/Transformer, False for XGBoost/RF
    
    // Performance tracking
    MultiOutputMetrics metrics_;
    std::vector<double> inferenceTimes_;
    
    // Helper methods
    bool loadMetadata();
    std::vector<float> normalizeInput(const std::vector<float>& input);
    void updateMetrics(double inferenceTime, const MultiOutputResult& result);
    
    // Model-specific prediction methods
    MultiOutputResult predictNeuralNetwork(const RadarSequence& sequence);
    MultiOutputResult predictXGBoost(const RadarSequence& sequence);
    MultiOutputResult predictRandomForest(const RadarSequence& sequence);
    
    // Parse output tensors into tags
    MultiOutputTags parseOutputTags(const std::vector<float>& outputs);
};

#endif // RADAR_TAGGER_MULTIOUTPUT_H

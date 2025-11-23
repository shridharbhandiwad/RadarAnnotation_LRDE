/**
 * @file radar_tagger.h
 * @brief Real-time radar trajectory tagger using TensorFlow Lite models
 */

#ifndef RADAR_TAGGER_H
#define RADAR_TAGGER_H

#include <string>
#include <vector>
#include <memory>
#include <map>

// Forward declarations for TensorFlow Lite
namespace tflite {
    class FlatBufferModel;
    class Interpreter;
    class InterpreterBuilder;
}

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
};

/**
 * @brief Model prediction result
 */
struct PredictionResult {
    int predictedClass;
    std::vector<float> classProbabilities;
    std::string className;
    double inferenceTimeMs;
    bool success;
    std::string errorMessage;
};

/**
 * @brief Performance metrics
 */
struct PerformanceMetrics {
    int totalInferences;
    double avgInferenceTimeMs;
    double minInferenceTimeMs;
    double maxInferenceTimeMs;
    double totalTimeMs;
    double throughput; // inferences per second
    
    void print() const;
};

/**
 * @brief Real-time Radar Tagger using TensorFlow Lite
 */
class RadarTagger {
public:
    /**
     * @brief Construct a new Radar Tagger
     * @param modelPath Path to TFLite model file
     * @param metadataPath Path to model metadata JSON file
     * @param numThreads Number of threads for inference (default: 4)
     */
    RadarTagger(const std::string& modelPath, 
                const std::string& metadataPath,
                int numThreads = 4);
    
    /**
     * @brief Destructor
     */
    ~RadarTagger();
    
    /**
     * @brief Initialize the model
     * @return true if successful, false otherwise
     */
    bool initialize();
    
    /**
     * @brief Predict class for a radar sequence
     * @param sequence Input radar sequence
     * @return Prediction result
     */
    PredictionResult predict(const RadarSequence& sequence);
    
    /**
     * @brief Batch prediction for multiple sequences
     * @param sequences Vector of radar sequences
     * @return Vector of prediction results
     */
    std::vector<PredictionResult> predictBatch(const std::vector<RadarSequence>& sequences);
    
    /**
     * @brief Load test data from CSV file
     * @param csvPath Path to CSV file
     * @return Vector of radar sequences
     */
    static std::vector<RadarSequence> loadFromCSV(const std::string& csvPath);
    
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
    PerformanceMetrics getMetrics() const;
    
    /**
     * @brief Reset performance metrics
     */
    void resetMetrics();
    
    /**
     * @brief Get class names
     */
    std::vector<std::string> getClassNames() const { return classNames_; }
    
    /**
     * @brief Get number of classes
     */
    int getNumClasses() const { return numClasses_; }

private:
    std::string modelPath_;
    std::string metadataPath_;
    int numThreads_;
    
    // TensorFlow Lite components
    std::unique_ptr<tflite::FlatBufferModel> model_;
    std::unique_ptr<tflite::Interpreter> interpreter_;
    
    // Model metadata
    std::vector<std::string> classNames_;
    std::vector<float> scalerMean_;
    std::vector<float> scalerScale_;
    std::vector<std::string> featureColumns_;
    int numClasses_;
    int sequenceLength_;
    int numFeatures_;
    
    // Performance tracking
    PerformanceMetrics metrics_;
    std::vector<double> inferenceTimes_;
    
    // Helper methods
    bool loadMetadata();
    std::vector<float> normalizeInput(const std::vector<float>& input);
    void updateMetrics(double inferenceTime);
};

#endif // RADAR_TAGGER_H

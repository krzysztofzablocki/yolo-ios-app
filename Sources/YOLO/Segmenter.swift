// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

//  This file is part of the Ultralytics YOLO Package, implementing instance segmentation functionality.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  The Segmenter class extends BasePredictor to provide instance segmentation capabilities.
//  Instance segmentation not only detects objects but also identifies the precise pixels
//  belonging to each object. The class processes complex model outputs including prototype masks
//  and detection results, performs non-maximum suppression to filter detections, and combines
//  results into visualizable mask images. It leverages the Accelerate framework for efficient
//  matrix operations and includes parallel processing to optimize performance on mobile devices.
//  The results include both bounding boxes and pixel-level masks that can be overlaid on images.

import Accelerate
@preconcurrency import CoreML
import Foundation
import UIKit
import Vision

/// Specialized predictor for YOLO segmentation models that identify objects and their pixel-level masks.
public class Segmenter: BasePredictor, @unchecked Sendable {
  /// Cached context reused by the single-image fast path.
  private let ciContext = CIContext(options: [
    .useSoftwareRenderer: false,
    .cacheIntermediates: false
  ])

  override func processObservations(for request: VNRequest, error: Error?) {
    if let results = request.results as? [VNCoreMLFeatureValueObservation] {
      guard results.count == 2 else { return }
      var pred: MLMultiArray
      var masks: MLMultiArray
      guard let out0 = results[0].featureValue.multiArrayValue,
        let out1 = results[1].featureValue.multiArrayValue
      else { return }
      let out0dim = checkShapeDimensions(of: out0)
      _ = checkShapeDimensions(of: out1)
      if out0dim == 4 {
        masks = out0
        pred = out1
      } else {
        masks = out1
        pred = out0
      }
      let detectedObjects = postProcessSegment(
        feature: pred, confidenceThreshold: Float(confidenceThreshold),
        iouThreshold: Float(iouThreshold))

      let detectionsCount = detectedObjects.count
      var boxes: [Box] = []
      boxes.reserveCapacity(detectionsCount)
      var alphas = [CGFloat]()
      alphas.reserveCapacity(detectionsCount)

      let modelWidth = CGFloat(self.modelInputSize.width)
      let modelHeight = CGFloat(self.modelInputSize.height)
      let inputWidth = Int(self.inputSize.width)
      let inputHeight = Int(self.inputSize.height)

      // Pre-calculate alpha constants
      let alphaScale: CGFloat = 0.9 / 0.8  // (1.0 - 0.2)
      let alphaOffset: CGFloat = -0.2 * alphaScale

      let limitedObjects = detectedObjects.prefix(self.numItemsThreshold)
      for p in limitedObjects {
        let box = p.0
        let rect = CGRect(
          x: box.minX / modelWidth, y: box.minY / modelHeight,
          width: box.width / modelWidth, height: box.height / modelHeight)
        let confidence = p.2
        let bestClass = p.1
        guard bestClass < self.labels.count else { continue }
        let label = self.labels[bestClass]
        let xywh = VNImageRectForNormalizedRect(rect, inputWidth, inputHeight)

        let boxResult = Box(index: bestClass, cls: label, conf: confidence, xywh: xywh, xywhn: rect)
        let alpha = CGFloat(confidence) * alphaScale + alphaOffset
        boxes.append(boxResult)
        alphas.append(alpha)
      }

      // Update timing before capturing values to avoid one-frame lag
      self.updateTime()

      // Capture needed values before async block
      let capturedMasks = masks
      let capturedBoxes = boxes
      let capturedInputSize = self.inputSize
      let capturedModelInputSize = self.modelInputSize
      let capturedT2 = self.t2
      let capturedT4 = self.t4
      let capturedLabels = self.labels

      let capturedDetectedObjects = Array(limitedObjects)
      DispatchQueue.global(qos: .userInitiated).async { [weak self] in
        guard
          let processedMasks = generateCombinedMaskImage(
            detectedObjects: capturedDetectedObjects,
            protos: capturedMasks,
            inputWidth: capturedModelInputSize.width,
            inputHeight: capturedModelInputSize.height,
            threshold: 0.5

          ) as? (CGImage?, [[[Float]]])
        else {
          DispatchQueue.main.async { [weak self] in
            self?.isUpdating = false
          }
          return
        }
        let maskResults = Masks(masks: processedMasks.1, combinedMask: processedMasks.0)
        let result = YOLOResult(
          orig_shape: capturedInputSize, boxes: capturedBoxes, masks: maskResults,
          speed: capturedT2,
          fps: 1 / capturedT4, names: capturedLabels)
        self?.currentOnResultsListener?.on(result: result)
      }
    }
  }

  private func updateTime() {
    if self.t1 < 10.0 {  // valid dt
      self.t2 = self.t1 * 0.05 + self.t2 * 0.95  // smoothed inference time
    }
    self.t4 = (CACurrentMediaTime() - self.t3) * 0.05 + self.t4 * 0.95  // smoothed delivered FPS
    self.t3 = CACurrentMediaTime()

    self.currentOnInferenceTimeListener?.on(inferenceTime: self.t2 * 1000, fpsRate: 1 / self.t4)  // t2 seconds to ms

  }

  public override func predictOnImage(image: CIImage) -> YOLOResult {
    let totalStart = CACurrentMediaTime()
    var timings: [(String, Double)] = []

    func logPhase(_ name: String, since start: CFTimeInterval) {
      let elapsed = (CACurrentMediaTime() - start) * 1000
      timings.append((name, elapsed))
    }

    let setupStart = CACurrentMediaTime()
    let requestHandler = VNImageRequestHandler(ciImage: image, options: [:])
    guard let request = visionRequest else {
      print("[Segment] ERROR: No vision request available")
      return YOLOResult(orig_shape: inputSize, boxes: [], speed: 0, names: labels)
    }

    let imageWidth = image.extent.width
    let imageHeight = image.extent.height
    self.inputSize = CGSize(width: imageWidth, height: imageHeight)
    logPhase("setup_handler_ciimage", since: setupStart)

    var result = YOLOResult(orig_shape: .zero, boxes: [], speed: 0, names: labels)

    let performStart = CACurrentMediaTime()
    do {
      try requestHandler.perform([request])
      logPhase("vision_perform", since: performStart)

      let parseStart = CACurrentMediaTime()
      guard let results = request.results as? [VNCoreMLFeatureValueObservation],
        results.count == 2
      else {
        print("[Segment] ERROR: Invalid results count: \(request.results?.count ?? 0)")
        printTimingSummary(timings, totalStart: totalStart, label: "Segment")
        return YOLOResult(orig_shape: .zero, boxes: [], speed: 0, names: labels)
      }

      guard
        let outputs = parseModelOutputs(results: results)
      else {
        print("[Segment] ERROR: Failed to parse model outputs")
        printTimingSummary(timings, totalStart: totalStart, label: "Segment")
        return YOLOResult(orig_shape: .zero, boxes: [], speed: 0, names: labels)
      }
      logPhase("parse_outputs", since: parseStart)

      let postProcessStart = CACurrentMediaTime()
      let detectedObjects = postProcessSegment(
        feature: outputs.pred,
        confidenceThreshold: Float(self.confidenceThreshold),
        iouThreshold: Float(self.iouThreshold)
      )
      logPhase("post_process_nms", since: postProcessStart)

      let boxesStart = CACurrentMediaTime()
      let boxes = makeBoxes(
        from: detectedObjects,
        outputSize: inputSize,
        maxCount: self.numItemsThreshold
      )
      logPhase("construct_boxes", since: boxesStart)

      let limitedObjects = Array(detectedObjects.prefix(self.numItemsThreshold))

      let masksStart = CACurrentMediaTime()
      guard
        let processedMasks = generateCombinedMaskImage(
          detectedObjects: limitedObjects,
          protos: outputs.masks,
          inputWidth: self.modelInputSize.width,
          inputHeight: self.modelInputSize.height,
          threshold: 0.5
        ) as? (CGImage?, [[[Float]]])
      else {
        logPhase("generate_masks", since: masksStart)
        printTimingSummary(timings, totalStart: totalStart, label: "Segment")
        return YOLOResult(
          orig_shape: inputSize,
          boxes: boxes,
          masks: nil,
          annotatedImage: nil,
          speed: 0,
          names: labels
        )
      }
      logPhase("generate_masks", since: masksStart)

      let resultStart = CACurrentMediaTime()
      let maskResults = Masks(masks: processedMasks.1, combinedMask: processedMasks.0)
      updateTime()

      result = YOLOResult(
        orig_shape: inputSize,
        boxes: boxes,
        masks: maskResults,
        annotatedImage: nil,
        speed: self.t2,
        fps: 1 / self.t4,
        names: labels
      )
      logPhase("construct_result", since: resultStart)

      printTimingSummary(timings, totalStart: totalStart, label: "Segment")
      return result
    } catch {
      logPhase("vision_perform_error", since: performStart)
      print("[Segment] ERROR: Vision perform failed: \(error)")
      printTimingSummary(timings, totalStart: totalStart, label: "Segment")
    }

    return result
  }

  /// Optimized single-image prediction that uses the CVPixelBuffer-based Vision path.
  public func predictOnImageFast(image: CIImage) -> YOLOResult {
    let totalStart = CACurrentMediaTime()
    var timings: [(String, Double)] = []

    func logPhase(_ name: String, since start: CFTimeInterval) {
      let elapsed = (CACurrentMediaTime() - start) * 1000
      timings.append((name, elapsed))
    }

    let setupStart = CACurrentMediaTime()
    guard let request = visionRequest else {
      print("[SegmentFast] ERROR: No vision request available")
      return YOLOResult(orig_shape: inputSize, boxes: [], speed: 0, names: labels)
    }

    let originalSize = CGSize(width: image.extent.width, height: image.extent.height)
    self.inputSize = originalSize
    let targetWidth = modelInputSize.width
    let targetHeight = modelInputSize.height
    logPhase("setup_validation", since: setupStart)

    let scalingStart = CACurrentMediaTime()
    let scaleX = CGFloat(targetWidth) / max(originalSize.width, 1)
    let scaleY = CGFloat(targetHeight) / max(originalSize.height, 1)
    let scaledImage = image.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
    logPhase("image_scaling", since: scalingStart)

    let bufferStart = CACurrentMediaTime()
    let attributes: [String: Any] = [
      kCVPixelBufferCGImageCompatibilityKey as String: true,
      kCVPixelBufferCGBitmapContextCompatibilityKey as String: true,
      kCVPixelBufferMetalCompatibilityKey as String: true
    ]

    var pixelBuffer: CVPixelBuffer?
    let status = CVPixelBufferCreate(
      kCFAllocatorDefault,
      targetWidth,
      targetHeight,
      kCVPixelFormatType_32BGRA,
      attributes as CFDictionary,
      &pixelBuffer
    )

    guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
      print("[SegmentFast] ERROR: Failed to create pixel buffer, status: \(status)")
      return YOLOResult(orig_shape: originalSize, boxes: [], speed: 0, names: labels)
    }
    logPhase("pixelbuffer_create", since: bufferStart)

    let renderStart = CACurrentMediaTime()
    ciContext.render(scaledImage, to: buffer)
    logPhase("cicontext_render", since: renderStart)

    let handlerStart = CACurrentMediaTime()
    let requestHandler = VNImageRequestHandler(cvPixelBuffer: buffer, options: [:])
    logPhase("handler_create", since: handlerStart)

    let performStart = CACurrentMediaTime()
    do {
      try requestHandler.perform([request])
      logPhase("vision_perform", since: performStart)

      let parseStart = CACurrentMediaTime()
      guard let results = request.results as? [VNCoreMLFeatureValueObservation],
        results.count == 2
      else {
        print("[SegmentFast] ERROR: Invalid results count: \(request.results?.count ?? 0)")
        printTimingSummary(timings, totalStart: totalStart, label: "SegmentFast")
        return YOLOResult(orig_shape: originalSize, boxes: [], speed: 0, names: labels)
      }

      guard
        let outputs = parseModelOutputs(results: results)
      else {
        print("[SegmentFast] ERROR: Failed to parse model outputs")
        printTimingSummary(timings, totalStart: totalStart, label: "SegmentFast")
        return YOLOResult(orig_shape: originalSize, boxes: [], speed: 0, names: labels)
      }
      logPhase("parse_outputs", since: parseStart)

      let postProcessStart = CACurrentMediaTime()
      let detectedObjects = postProcessSegment(
        feature: outputs.pred,
        confidenceThreshold: Float(self.confidenceThreshold),
        iouThreshold: Float(self.iouThreshold)
      )
      logPhase("post_process_nms", since: postProcessStart)

      let boxesStart = CACurrentMediaTime()
      let boxes = makeBoxes(
        from: detectedObjects,
        outputSize: originalSize,
        maxCount: self.numItemsThreshold
      )
      logPhase("construct_boxes", since: boxesStart)

      let limitedObjects = Array(detectedObjects.prefix(self.numItemsThreshold))

      let masksStart = CACurrentMediaTime()
      guard
        let processedMasks = generateCombinedMaskImage(
          detectedObjects: limitedObjects,
          protos: outputs.masks,
          inputWidth: self.modelInputSize.width,
          inputHeight: self.modelInputSize.height,
          threshold: 0.5
        ) as? (CGImage?, [[[Float]]])
      else {
        logPhase("generate_masks", since: masksStart)
        printTimingSummary(timings, totalStart: totalStart, label: "SegmentFast")
        return YOLOResult(
          orig_shape: originalSize,
          boxes: boxes,
          masks: nil,
          annotatedImage: nil,
          speed: 0,
          names: labels
        )
      }
      logPhase("generate_masks", since: masksStart)

      let resultStart = CACurrentMediaTime()
      let maskResults = Masks(masks: processedMasks.1, combinedMask: processedMasks.0)
      updateTime()

      let result = YOLOResult(
        orig_shape: originalSize,
        boxes: boxes,
        masks: maskResults,
        annotatedImage: nil,
        speed: self.t2,
        fps: 1 / self.t4,
        names: labels
      )
      logPhase("construct_result", since: resultStart)

      printTimingSummary(timings, totalStart: totalStart, label: "SegmentFast")
      return result
    } catch {
      logPhase("vision_perform_error", since: performStart)
      print("[SegmentFast] ERROR: Vision perform failed: \(error)")
      printTimingSummary(timings, totalStart: totalStart, label: "SegmentFast")
      return YOLOResult(orig_shape: originalSize, boxes: [], speed: 0, names: labels)
    }
  }

  private func parseModelOutputs(results: [VNCoreMLFeatureValueObservation]) -> (
    pred: MLMultiArray, masks: MLMultiArray
  )? {
    guard let out0 = results[0].featureValue.multiArrayValue,
      let out1 = results[1].featureValue.multiArrayValue
    else {
      return nil
    }

    let out0dim = checkShapeDimensions(of: out0)
    _ = checkShapeDimensions(of: out1)
    if out0dim == 4 {
      return (pred: out1, masks: out0)
    }

    return (pred: out0, masks: out1)
  }

  private func makeBoxes(
    from detectedObjects: [(CGRect, Int, Float, MLMultiArray)],
    outputSize: CGSize,
    maxCount: Int
  ) -> [Box] {
    let limitedObjects = detectedObjects.prefix(maxCount)
    let modelWidth = CGFloat(self.modelInputSize.width)
    let modelHeight = CGFloat(self.modelInputSize.height)
    let inputWidth = Int(outputSize.width)
    let inputHeight = Int(outputSize.height)

    var boxes: [Box] = []
    boxes.reserveCapacity(limitedObjects.count)

    for prediction in limitedObjects {
      let box = prediction.0
      let rect = CGRect(
        x: box.minX / modelWidth,
        y: box.minY / modelHeight,
        width: box.width / modelWidth,
        height: box.height / modelHeight
      )
      let confidence = prediction.2
      let bestClass = prediction.1
      guard bestClass < labels.count else { continue }
      let label = labels[bestClass]
      let xywh = VNImageRectForNormalizedRect(rect, inputWidth, inputHeight)

      boxes.append(Box(index: bestClass, cls: label, conf: confidence, xywh: xywh, xywhn: rect))
    }

    return boxes
  }

  private func printTimingSummary(
    _ timings: [(String, Double)],
    totalStart: CFTimeInterval,
    label: String
  ) {
    let totalMs = max((CACurrentMediaTime() - totalStart) * 1000, .ulpOfOne)
    print("[\(label)] ----------------------------------------")
    print("[\(label)] Timing breakdown (ms)")
    for (phase, milliseconds) in timings {
      let percentage = (milliseconds / totalMs) * 100
      let msText = String(format: "%.2f", milliseconds)
      let percentageText = String(format: "%.1f", percentage)
      print("[\(label)] \(phase): \(msText) ms (\(percentageText)%)")
    }
    print("[\(label)] Total: \(String(format: "%.2f", totalMs)) ms")
    print("[\(label)] ----------------------------------------")
  }

  nonisolated func postProcessSegment(
    feature: MLMultiArray,
    confidenceThreshold: Float,
    iouThreshold: Float
  ) -> [(CGRect, Int, Float, MLMultiArray)] {
    let shape = feature.shape.map { $0.intValue }
    guard shape.count == 3 else { return [] }

    // YOLO26 end2end seg: [1, max_det, 6+32] where shape[2] < shape[1]
    // Traditional seg: [1, 4+nc+32, num_anchors] where shape[2] > shape[1]
    if shape[2] < shape[1] {
      return postProcessEnd2EndSegment(
        feature: feature, shape: shape, confidenceThreshold: confidenceThreshold)
    }

    let numAnchors = shape[2]
    let numFeatures = shape[1]
    let boxFeatureLength = 4
    let maskConfidenceLength = 32
    let numClasses = numFeatures - boxFeatureLength - maskConfidenceLength

    // Pre-allocate result arrays with estimated capacity
    let estimatedCapacity = min(numAnchors / 10, 100)  // Estimate ~10% detection rate
    let resultsWrapper = ResultsWrapper(capacity: estimatedCapacity)

    // Wrapper for thread-safe results collection
    final class ResultsWrapper: @unchecked Sendable {
      private let lock = NSLock()
      private var results: [(CGRect, Int, Float, MLMultiArray)]

      init(capacity: Int) {
        self.results = []
        self.results.reserveCapacity(capacity)
      }

      func append(_ result: (CGRect, Int, Float, MLMultiArray)) {
        lock.lock()
        results.append(result)
        lock.unlock()
      }

      func getResults() -> [(CGRect, Int, Float, MLMultiArray)] {
        return results
      }
    }

    let featurePointer = feature.dataPointer.assumingMemoryBound(to: Float.self)
    let pointerWrapper = FloatPointerWrapper(featurePointer)

    DispatchQueue.concurrentPerform(iterations: numAnchors) { j in
      let x = pointerWrapper.pointer[j]
      let y = pointerWrapper.pointer[numAnchors + j]
      let width = pointerWrapper.pointer[2 * numAnchors + j]
      let height = pointerWrapper.pointer[3 * numAnchors + j]

      let boxX = CGFloat(x - width / 2)
      let boxY = CGFloat(y - height / 2)
      let boundingBox = CGRect(x: boxX, y: boxY, width: CGFloat(width), height: CGFloat(height))

      // Use thread-local storage for class probabilities
      let localClassProbs = UnsafeMutableBufferPointer<Float>.allocate(capacity: numClasses)
      defer { localClassProbs.deallocate() }

      vDSP_mtrans(
        pointerWrapper.pointer + 4 * numAnchors + j,
        numAnchors,
        localClassProbs.baseAddress!,
        1,
        1,
        vDSP_Length(numClasses)
      )

      var maxClassValue: Float = 0
      var maxClassIndex: vDSP_Length = 0
      vDSP_maxvi(
        localClassProbs.baseAddress!, 1, &maxClassValue, &maxClassIndex, vDSP_Length(numClasses))

      if maxClassValue > confidenceThreshold {
        // Create MLMultiArray more efficiently
        guard
          let maskProbs = try? MLMultiArray(
            shape: [NSNumber(value: maskConfidenceLength)], dataType: .float32)
        else {
          return
        }

        let maskProbsPointer = pointerWrapper.pointer + (4 + numClasses) * numAnchors + j
        let maskProbsData = maskProbs.dataPointer.assumingMemoryBound(to: Float.self)

        for i in 0..<maskConfidenceLength {
          maskProbsData[i] = maskProbsPointer[i * numAnchors]
        }

        let result = (boundingBox, Int(maxClassIndex), maxClassValue, maskProbs)

        resultsWrapper.append(result)

      }
    }

    // Get results from wrapper
    let collectedResults = resultsWrapper.getResults()

    // Optimize NMS by grouping results by class first
    var classBuckets: [Int: [(CGRect, Int, Float, MLMultiArray)]] = [:]
    for result in collectedResults {
      let classIndex = result.1
      if classBuckets[classIndex] == nil {
        classBuckets[classIndex] = []
        classBuckets[classIndex]!.reserveCapacity(collectedResults.count / numClasses + 1)
      }
      classBuckets[classIndex]?.append(result)
    }

    var selectedBoxesAndFeatures: [(CGRect, Int, Float, MLMultiArray)] = []
    selectedBoxesAndFeatures.reserveCapacity(collectedResults.count)

    for (_, classResults) in classBuckets {
      let boxesOnly = classResults.map { $0.0 }
      let scoresOnly = classResults.map { $0.2 }
      let selectedIndices = nonMaxSuppression(
        boxes: boxesOnly,
        scores: scoresOnly,
        threshold: iouThreshold
      )
      for idx in selectedIndices {
        selectedBoxesAndFeatures.append(classResults[idx])
      }
    }

    return selectedBoxesAndFeatures
  }

  /// Processes YOLO26 end2end segmentation output: [1, max_det, 6+32].
  /// Each detection: [x1, y1, x2, y2, conf, class_id, mask_0...mask_31] in xyxy pixel coords.
  /// NMS is already applied by the model, so no additional NMS is needed.
  private nonisolated func postProcessEnd2EndSegment(
    feature: MLMultiArray,
    shape: [Int],
    confidenceThreshold: Float
  ) -> [(CGRect, Int, Float, MLMultiArray)] {
    let numDetections = shape[1]
    let numFields = shape[2]
    let maskCoefficients = 32
    let strides = feature.strides.map { $0.intValue }
    let pointer = feature.dataPointer.assumingMemoryBound(to: Float.self)
    let detStride = strides[1]
    let fieldStride = strides[2]

    var results: [(CGRect, Int, Float, MLMultiArray)] = []

    for i in 0..<numDetections {
      let base = i * detStride
      let conf = pointer[base + 4 * fieldStride]
      guard conf > confidenceThreshold else { continue }

      let x1 = CGFloat(pointer[base])
      let y1 = CGFloat(pointer[base + fieldStride])
      let x2 = CGFloat(pointer[base + 2 * fieldStride])
      let y2 = CGFloat(pointer[base + 3 * fieldStride])
      let classId = numFields > 5 ? Int(pointer[base + 5 * fieldStride]) : 0

      let boundingBox = CGRect(x: x1, y: y1, width: x2 - x1, height: y2 - y1)

      // Extract mask coefficients (fields 6..37)
      guard
        let maskProbs = try? MLMultiArray(
          shape: [NSNumber(value: maskCoefficients)], dataType: .float32)
      else { continue }
      let maskProbsData = maskProbs.dataPointer.assumingMemoryBound(to: Float.self)
      let maskStartField = numFields > 5 ? 6 : 5
      for m in 0..<min(maskCoefficients, numFields - maskStartField) {
        maskProbsData[m] = pointer[base + (maskStartField + m) * fieldStride]
      }

      results.append((boundingBox, classId, conf, maskProbs))
    }

    return results
  }

  func checkShapeDimensions(of multiArray: MLMultiArray) -> Int {
    let shapeAsInts = multiArray.shape.map { $0.intValue }
    let dimensionCount = shapeAsInts.count

    return dimensionCount
  }

}

final class FloatPointerWrapper: @unchecked Sendable {
  let pointer: UnsafeMutablePointer<Float>
  init(_ pointer: UnsafeMutablePointer<Float>) {
    self.pointer = pointer
  }
}

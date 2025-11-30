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
  var colorsForMask: [(red: UInt8, green: UInt8, blue: UInt8)] = []

  /// Cached CIContext for efficient image rendering (reused across calls)
  private let ciContext = CIContext(options: [
    .useSoftwareRenderer: false,
    .cacheIntermediates: false
  ])

  override func processObservations(for request: VNRequest, error: Error?) {
    if let results = request.results as? [VNCoreMLFeatureValueObservation] {
      //            DispatchQueue.main.async { [self] in
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
        let label = self.labels[bestClass]
        let xywh = VNImageRectForNormalizedRect(rect, inputWidth, inputHeight)

        let boxResult = Box(index: bestClass, cls: label, conf: confidence, xywh: xywh, xywhn: rect)
        let alpha = CGFloat(confidence) * alphaScale + alphaOffset
        boxes.append(boxResult)
        alphas.append(alpha)
      }

      // Capture needed values before async block
      let capturedMasks = masks
      let capturedBoxes = boxes
      let capturedInputSize = self.inputSize
      let capturedModelInputSize = self.modelInputSize
      let capturedT2 = self.t2
      let capturedT4 = self.t4
      let capturedLabels = self.labels

      DispatchQueue.global(qos: .userInitiated).async { [weak self] in
        guard
          let processedMasks = generateCombinedMaskImage(
            detectedObjects: detectedObjects,
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
        self?.updateTime()
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

    // Phase 1: Setup and create handler with CIImage
    let phase1Start = CACurrentMediaTime()
    let requestHandler = VNImageRequestHandler(ciImage: image, options: [:])
    guard let request = visionRequest else {
      let emptyResult = YOLOResult(orig_shape: inputSize, boxes: [], speed: 0, names: labels)
      return emptyResult
    }

    let imageWidth = image.extent.width
    let imageHeight = image.extent.height
    self.inputSize = CGSize(width: imageWidth, height: imageHeight)
    logPhase("1_setup_handler_ciimage", since: phase1Start)

    var result = YOLOResult(orig_shape: .zero, boxes: [], speed: 0, names: labels)

    // Phase 2: Perform Vision request
    let phase2Start = CACurrentMediaTime()
    do {
      try requestHandler.perform([request])
      logPhase("2_vision_perform", since: phase2Start)

      // Phase 3: Parse model outputs
      let phase3Start = CACurrentMediaTime()
      guard let results = request.results as? [VNCoreMLFeatureValueObservation],
            results.count == 2 else {
        print("[Segment] ERROR: Invalid results count")
        return YOLOResult(orig_shape: .zero, boxes: [], speed: 0, names: labels)
      }

      var pred: MLMultiArray
      var masks: MLMultiArray
      guard let out0 = results[0].featureValue.multiArrayValue,
            let out1 = results[1].featureValue.multiArrayValue else {
        return YOLOResult(orig_shape: .zero, boxes: [], speed: 0, names: labels)
      }

      let out0dim = checkShapeDimensions(of: out0)
      _ = checkShapeDimensions(of: out1)
      if out0dim == 4 {
        masks = out0
        pred = out1
      } else {
        masks = out1
        pred = out0
      }
      logPhase("3_parse_outputs", since: phase3Start)

      // Phase 4: Post-process detection results
      let phase4Start = CACurrentMediaTime()
      let detectedObjects = postProcessSegment(
        feature: pred, confidenceThreshold: Float(self.confidenceThreshold),
        iouThreshold: Float(self.iouThreshold))
      logPhase("4_post_process_nms", since: phase4Start)

      // Phase 5: Construct bounding box information
      let phase5Start = CACurrentMediaTime()
      let detectionsCount = detectedObjects.count
      var boxes: [Box] = []
      boxes.reserveCapacity(detectionsCount)

      let modelWidth = CGFloat(self.modelInputSize.width)
      let modelHeight = CGFloat(self.modelInputSize.height)
      let inputWidth = Int(inputSize.width)
      let inputHeight = Int(inputSize.height)

      let limitedObjects = detectedObjects.prefix(self.numItemsThreshold)
      for p in limitedObjects {
        let box = p.0
        let rect = CGRect(
          x: box.minX / modelWidth, y: box.minY / modelHeight,
          width: box.width / modelWidth, height: box.height / modelHeight)
        let confidence = p.2
        let bestClass = p.1
        let label = labels[bestClass]
        let xywh = VNImageRectForNormalizedRect(rect, inputWidth, inputHeight)

        let boxResult = Box(
          index: bestClass, cls: label, conf: confidence, xywh: xywh, xywhn: rect)
        boxes.append(boxResult)
      }
      logPhase("5_construct_boxes", since: phase5Start)

      // Phase 6: Generate mask image
      let phase6Start = CACurrentMediaTime()
      guard
        let processedMasks = generateCombinedMaskImage(
          detectedObjects: Array(limitedObjects),
          protos: masks,
          inputWidth: self.modelInputSize.width,
          inputHeight: self.modelInputSize.height,
          threshold: 0.5
        ) as? (CGImage?, [[[Float]]])
      else {
        logPhase("6_generate_masks", since: phase6Start)
        printTimingSummary(timings, totalStart: totalStart, label: "Segment")
        return YOLOResult(
          orig_shape: inputSize, boxes: boxes, masks: nil, annotatedImage: nil, speed: 0,
          names: labels)
      }
      logPhase("6_generate_masks", since: phase6Start)

      // Phase 7: Draw annotations
//      let phase7Start = CACurrentMediaTime()
//      let annotatedImage = drawYOLOSegmentationWithBoxes(
//        ciImage: image,
//        boxes: boxes,
//        maskImage: processedMasks.0,
//        originalImageSize: inputSize
//      )
//      logPhase("7_draw_annotations", since: phase7Start)

      // Phase 8: Construct result
      let phase8Start = CACurrentMediaTime()
      let maskResults: Masks = Masks(masks: processedMasks.1, combinedMask: processedMasks.0)
      updateTime()

      result = YOLOResult(
        orig_shape: inputSize,
        boxes: boxes,
        masks: maskResults,
        annotatedImage: nil,//annotatedImage,
        speed: self.t2,
        fps: 1 / self.t4,
        names: labels
      )
      logPhase("8_construct_result", since: phase8Start)

      // Print timing summary
      printTimingSummary(timings, totalStart: totalStart, label: "Segment")

      return result

    } catch {
      logPhase("2_vision_perform_ERROR", since: phase2Start)
      print("[Segment] ERROR: Vision perform failed: \(error)")
      printTimingSummary(timings, totalStart: totalStart, label: "Segment")
    }
    return result
  }

  // MARK: - Fast Single Image Prediction

  /// Optimized single-image prediction using CVPixelBuffer instead of CIImage for faster Vision processing.
  /// Includes detailed timing logs for performance analysis.
  ///
  /// - Parameter image: The CIImage to process.
  /// - Returns: A YOLOResult containing the prediction outputs.
  public func predictOnImageFast(image: CIImage) -> YOLOResult {
    let totalStart = CACurrentMediaTime()
    var timings: [(String, Double)] = []

    func logPhase(_ name: String, since start: CFTimeInterval) {
      let elapsed = (CACurrentMediaTime() - start) * 1000
      timings.append((name, elapsed))
    }

    // Phase 1: Setup and validation
    let phase1Start = CACurrentMediaTime()
    guard let request = visionRequest else {
      print("[SegmentFast] ERROR: No vision request available")
      return YOLOResult(orig_shape: inputSize, boxes: [], speed: 0, names: labels)
    }

    let originalWidth = image.extent.width
    let originalHeight = image.extent.height
    self.inputSize = CGSize(width: originalWidth, height: originalHeight)

    let targetWidth = modelInputSize.width
    let targetHeight = modelInputSize.height
    logPhase("1_setup_validation", since: phase1Start)

    // Phase 2: Scale image to model input size
    let phase2Start = CACurrentMediaTime()
    let scaleX = CGFloat(targetWidth) / originalWidth
    let scaleY = CGFloat(targetHeight) / originalHeight
    let scaledImage = image.transformed(by: CGAffineTransform(scaleX: scaleX, y: scaleY))
    logPhase("2_image_scaling", since: phase2Start)

    // Phase 3: Create CVPixelBuffer
    let phase3Start = CACurrentMediaTime()
    let attrs: [String: Any] = [
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
      attrs as CFDictionary,
      &pixelBuffer
    )

    guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
      print("[SegmentFast] ERROR: Failed to create pixel buffer, status: \(status)")
      return YOLOResult(orig_shape: inputSize, boxes: [], speed: 0, names: labels)
    }
    logPhase("3_pixelbuffer_create", since: phase3Start)

    // Phase 4: Render CIImage to CVPixelBuffer
    let phase4Start = CACurrentMediaTime()
    ciContext.render(scaledImage, to: buffer)
    logPhase("4_cicontext_render", since: phase4Start)

    // Phase 5: Create Vision request handler
    let phase5Start = CACurrentMediaTime()
    let requestHandler = VNImageRequestHandler(cvPixelBuffer: buffer, options: [:])
    logPhase("5_handler_create", since: phase5Start)

    // Phase 6: Perform Vision request (the main inference)
    let phase6Start = CACurrentMediaTime()
    var result = YOLOResult(orig_shape: .zero, boxes: [], speed: 0, names: labels)

    do {
      try requestHandler.perform([request])
      logPhase("6_vision_perform", since: phase6Start)

      // Phase 7: Parse model outputs
      let phase7Start = CACurrentMediaTime()
      guard let results = request.results as? [VNCoreMLFeatureValueObservation],
            results.count == 2 else {
        print("[SegmentFast] ERROR: Invalid results count: \(request.results?.count ?? 0)")
        return YOLOResult(orig_shape: .zero, boxes: [], speed: 0, names: labels)
      }

      var pred: MLMultiArray
      var masks: MLMultiArray
      guard let out0 = results[0].featureValue.multiArrayValue,
            let out1 = results[1].featureValue.multiArrayValue else {
        print("[SegmentFast] ERROR: Failed to get multiArrayValue from results")
        return YOLOResult(orig_shape: .zero, boxes: [], speed: 0, names: labels)
      }

      let out0dim = checkShapeDimensions(of: out0)
      if out0dim == 4 {
        masks = out0
        pred = out1
      } else {
        masks = out1
        pred = out0
      }
      logPhase("7_parse_outputs", since: phase7Start)

      // Phase 8: Post-process detections (NMS, etc.)
      let phase8Start = CACurrentMediaTime()
      let detectedObjects = postProcessSegment(
        feature: pred,
        confidenceThreshold: Float(self.confidenceThreshold),
        iouThreshold: Float(self.iouThreshold)
      )
      logPhase("8_post_process_nms", since: phase8Start)

      // Phase 9: Construct bounding boxes
      let phase9Start = CACurrentMediaTime()
      let detectionsCount = detectedObjects.count
      var boxes: [Box] = []
      boxes.reserveCapacity(detectionsCount)

      let modelWidth = CGFloat(self.modelInputSize.width)
      let modelHeight = CGFloat(self.modelInputSize.height)
      let inputWidth = Int(inputSize.width)
      let inputHeight = Int(inputSize.height)

      let limitedObjects = detectedObjects.prefix(self.numItemsThreshold)
      for p in limitedObjects {
        let box = p.0
        let rect = CGRect(
          x: box.minX / modelWidth, y: box.minY / modelHeight,
          width: box.width / modelWidth, height: box.height / modelHeight
        )
        let confidence = p.2
        let bestClass = p.1
        let label = labels[bestClass]
        let xywh = VNImageRectForNormalizedRect(rect, inputWidth, inputHeight)

        let boxResult = Box(
          index: bestClass, cls: label, conf: confidence, xywh: xywh, xywhn: rect
        )
        boxes.append(boxResult)
      }
      logPhase("9_construct_boxes", since: phase9Start)

      // Phase 10: Generate mask image
      let phase10Start = CACurrentMediaTime()
      guard let processedMasks = generateCombinedMaskImage(
        detectedObjects: Array(limitedObjects),
        protos: masks,
        inputWidth: self.modelInputSize.width,
        inputHeight: self.modelInputSize.height,
        threshold: 0.5
      ) as? (CGImage?, [[[Float]]]) else {
        logPhase("10_generate_masks", since: phase10Start)
        printTimingSummary(timings, totalStart: totalStart)
        return YOLOResult(
          orig_shape: inputSize, boxes: boxes, masks: nil, annotatedImage: nil, speed: 0,
          names: labels
        )
      }
      logPhase("10_generate_masks", since: phase10Start)

//      // Phase 11: Draw annotations
//      let phase11Start = CACurrentMediaTime()
//      let annotatedImage = drawYOLOSegmentationWithBoxes(
//        ciImage: image,
//        boxes: boxes,
//        maskImage: processedMasks.0,
//        originalImageSize: inputSize
//      )
//      logPhase("11_draw_annotations", since: phase11Start)

      // Phase 12: Construct final result
      let phase12Start = CACurrentMediaTime()
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
      logPhase("12_construct_result", since: phase12Start)

      // Print timing summary
      printTimingSummary(timings, totalStart: totalStart)

      return result

    } catch {
      logPhase("6_vision_perform_ERROR", since: phase6Start)
      print("[SegmentFast] ERROR: Vision perform failed: \(error)")
      printTimingSummary(timings, totalStart: totalStart)
    }

    return result
  }

  /// Prints a formatted timing summary to console
  private func printTimingSummary(_ timings: [(String, Double)], totalStart: CFTimeInterval, label: String = "SegmentFast") {
    let totalMs = (CACurrentMediaTime() - totalStart) * 1000
    print("\n[\(label)] ═══════════════════════════════════════")
    print("[\(label)] TIMING BREAKDOWN (ms):")
    print("[\(label)] ───────────────────────────────────────")
    for (phase, ms) in timings {
      let percentage = (ms / totalMs) * 100
      let bar = String(repeating: "█", count: Int(percentage / 5))
      print(String(format: "[\(label)] %-25s %7.2f ms (%5.1f%%) %@", (phase as NSString).utf8String!, ms, percentage, bar))
    }
    print("[\(label)] ───────────────────────────────────────")
    print(String(format: "[\(label)] TOTAL:                    %7.2f ms", totalMs))
    print("[\(label)] ═══════════════════════════════════════\n")
  }

  nonisolated func postProcessSegment(
    feature: MLMultiArray,
    confidenceThreshold: Float,
    iouThreshold: Float
  ) -> [(CGRect, Int, Float, MLMultiArray)] {

    let numAnchors = feature.shape[2].intValue
    let numFeatures = feature.shape[1].intValue
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

    // Pre-allocate reusable arrays outside the loop
    let classProbs = UnsafeMutableBufferPointer<Float>.allocate(capacity: numClasses)
    defer { classProbs.deallocate() }

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

final class ResultsWrapper: @unchecked Sendable {
  private var results: [(CGRect, Int, Float, MLMultiArray)] = []

  func reserveCapacity(_ capacity: Int) {
    results.reserveCapacity(capacity)
  }

  func append(_ result: (CGRect, Int, Float, MLMultiArray)) {
    results.append(result)
  }

  func getResults() -> [(CGRect, Int, Float, MLMultiArray)] {
    return results
  }
}

using System;
using System.Collections.Generic;
using System.Linq;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Filtering.Kalman;

namespace SORT_CSharp
{
    public class Sort
    {
        private int frameCount = 0;
        private List<KalmanBoxTracker> trackers = new List<KalmanBoxTracker>();
        private double iouThreshold = 0.3;
        private int maxAge = 1;
        private int minHits = 3;

        public Sort(int maxAge = 1, int minHits = 3, double iouThreshold = 0.3)
        {
            this.maxAge = maxAge;
            this.minHits = minHits;
            this.iouThreshold = iouThreshold;
        }

        public double[][] Update(double[][] dets)
        {
            frameCount++;
            List<double[]> ret = new List<double[]>();
            List<int> toDel = new List<int>();

            // Predict new locations for existing trackers
            for (int t = 0; t < trackers.Count; t++)
            {
                double[] pos = trackers[t].Predict();
                if (pos.Any(double.IsNaN))
                {
                    toDel.Add(t);
                }
            }

            toDel.Reverse();
            foreach (int t in toDel)
            {
                trackers.RemoveAt(t);
            }

            // Associating detections to tracked objects
            (List<int[]> matches, List<int> unmatchedDetections, List<int> unmatchedTrackers) = AssociateDetectionsToTrackers(dets, trackers, iouThreshold);

            // Update matched trackers with assigned detections
            foreach (var match in matches)
            {
                trackers[match[1]].Update(dets[match[0]]);
            }

            // Create and initialise new trackers for unmatched detections
            foreach (var i in unmatchedDetections)
            {
                var trk = new KalmanBoxTracker(dets[i]);
                trackers.Add(trk);
            }

            // Update tracker states
            for (int i = trackers.Count - 1; i >= 0; i--)
            {
                var trk = trackers[i];
                var d = trk.GetState();
                if (trk.TimeSinceUpdate < 1 && (trk.HitStreak >= minHits || frameCount <= minHits))
                {
                    ret.Add(new double[] { d[0], d[1], d[2], d[3], trk.Id });  // Return a tracker's state
                }
                // Remove dead tracklet
                if (trk.TimeSinceUpdate > maxAge)
                {
                    trackers.RemoveAt(i);
                }
            }

            return ret.ToArray();
        }

        // Convert bbox to z
        public static Vector<float> ConvertBboxToZ(Vector<float> bbox)
        {
            var width = bbox[2] - bbox[0];
            var height = bbox[3] - bbox[1];
            var x = bbox[0] + width / 2;
            var y = bbox[1] + height / 2;
            var s = width * height;    //scale is area
            var r = width / height;
            var z = Vector<float>.Build.DenseOfArray(new float[] { x, y, s, r });
            return z;
        }

        // Convert x to bbox
        public static Vector<float> ConvertXToBbox(Vector<float> xState)
        {
            var w = (float)Math.Sqrt(xState[2] * xState[3]);
            var h = (float)(xState[2] / w);
            var x = xState[0] - w / 2;
            var y = xState[1] - h / 2;
            var bbox = Vector<float>.Build.DenseOfArray(new float[] { x, y, x + w, y + h });
            return bbox;
        }

        // IoU Calculation for two boxes
        public static float IoU(Vector<float> boxA, Vector<float> boxB)
        {
            // Determine the coordinates of the intersection rectangle
            float xA = Math.Max(boxA[0], boxB[0]);
            float yA = Math.Max(boxA[1], boxB[1]);
            float xB = Math.Min(boxA[2], boxB[2]);
            float yB = Math.Min(boxA[3], boxB[3]);

            // Compute the area of intersection rectangle
            float interArea = Math.Max(0, xB - xA + 1) * Math.Max(0, yB - yA + 1);

            // Compute the area of both the prediction and ground-truth rectangles
            float boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1);
            float boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1);

            // Compute the intersection over union
            float iou = interArea / (boxAArea + boxBArea - interArea);

            return iou;
        }

        // IoU Batch
        public static Matrix<float> IoUBatch(double[][] detections, List<KalmanBoxTracker> trackers)
        {
            int dCount = detections.Length;
            int tCount = trackers.Count;

            Matrix<float> iouMatrix = Matrix<float>.Build.Dense(dCount, tCount, 0.0f);

            for (int i = 0; i < dCount; i++)
            {
                for (int j = 0; j < tCount; j++)
                {
                    Vector<float> detBox = Vector<float>.Build.DenseOfArray(detections[i].Select(x => (float)x).ToArray());
                    iouMatrix[i, j] = IoU(detBox, trackers[j].GetState());
                }
            }

            return iouMatrix;
        }

        public static (List<int>, List<int>) LinearAssignment(Matrix<float> costMatrix)
        {
            Matrix<float> costMatrixClone = costMatrix.Clone();
            int n = costMatrixClone.RowCount;
            int m = costMatrixClone.ColumnCount;

            int[] indices = Enumerable.Range(0, Math.Max(n, m)).ToArray();
            List<int> resultRows = new List<int>();
            List<int> resultCols = new List<int>();

            if (n <= m)
            {
                while (costMatrixClone.RowCount > 0)
                {
                    var minInColumns = Enumerable.Range(0, m).Select(i => costMatrixClone.Column(i).Minimum()).ToList();
                    var minValuesMatrix = Matrix<float>.Build.DenseOfColumnMajor(costMatrixClone.RowCount, costMatrixClone.ColumnCount, minInColumns);
                    costMatrixClone = costMatrixClone - minValuesMatrix;

                    var minInRows = Enumerable.Range(0, n).Select(i => costMatrixClone.Row(i).Minimum()).ToList();
                    var minValuesMatrixRow = Matrix<float>.Build.DenseOfRowMajor(costMatrixClone.RowCount, costMatrixClone.ColumnCount, minInRows);
                    costMatrixClone = costMatrixClone - minValuesMatrixRow;

                    var lines = new HashSet<int>();

                    while (lines.Count < costMatrixClone.RowCount)
                    {
                        var zeroInRow = costMatrixClone.Find(x => x == 0.0f);
                        while (zeroInRow.Item1 != -1)
                        {
                            lines.Add(zeroInRow.Item1);
                            zeroInRow = costMatrixClone.Find(x => x == 0.0f);
                        }

                        if (lines.Count >= costMatrixClone.RowCount)
                            break;

                        var h = float.PositiveInfinity;
                        for (int i = 0; i < costMatrixClone.RowCount; i++)
                        {
                            if (!lines.Contains(i))
                                h = Math.Min(h, costMatrixClone.Row(i).Minimum());
                        }

                        for (int i = 0; i < costMatrixClone.RowCount; i++)
                        {
                            if (lines.Contains(i))
                                costMatrixClone.SetRow(i, costMatrixClone.Row(i) + h);
                            else
                                costMatrixClone.SetRow(i, costMatrixClone.Row(i) - h);
                        }
                    }

                    var mask = costMatrixClone.Map2((a, b) => a == b ? 1.0f : 0.0f, costMatrixClone.Clone());
                    var rowInd = mask.Row(0).ToList().FindIndex(x => x != 0.0f);
                    var colInd = mask.Column(0).ToList().FindIndex(x => x != 0.0f);


                    resultRows.Add(indices[rowInd]);
                    resultCols.Add(indices[colInd]);

                    costMatrixClone = costMatrixClone.RemoveColumn(0);
                    costMatrixClone = costMatrixClone.RemoveRow(0);

                    indices = indices.Skip(1).ToArray();
                }
            }
            else
            {
                var transposeCostMatrix = costMatrixClone.Transpose();
                var assignmentResult = LinearAssignment(transposeCostMatrix);
                resultRows = assignmentResult.Item2;
                resultCols = assignmentResult.Item1;
            }

            return (resultRows, resultCols);
        }

        // Associate detections to tracked objects
        public static (List<int[]>, List<int>, List<int>) AssociateDetectionsToTrackers(double[][] detections, List<KalmanBoxTracker> trackers, double iouThreshold = 0.3f)
        {
            // Get IoU matrix
            var iouMatrix = IoUBatch(detections, trackers);

            List<int[]> matches = new List<int[]>();
            List<int> unmatchedDetections = new List<int>();
            List<int> unmatchedTrackers = new List<int>();

            // If no trackers, return empty matched and unmatched lists
            if (trackers.Count == 0)
            {
                for (int i = 0; i < detections.Length; i++)
                    unmatchedDetections.Add(i);
                return (matches, unmatchedDetections, unmatchedTrackers);
            }

            int dCount = detections.Length;
            int tCount = trackers.Count;

            List<int> detectionIndices = Enumerable.Range(0, dCount).ToList();
            List<int> trackerIndices = Enumerable.Range(0, tCount).ToList();

            // Perform linear assignment
            var (matchedIndicesD, matchedIndicesT) = LinearAssignment(-iouMatrix);

            // Map matched indices to original detection and tracker indices
            for (int m = 0; m < matchedIndicesD.Count; m++)
            {
                if (iouMatrix[matchedIndicesD[m], matchedIndicesT[m]] < iouThreshold)
                {
                    unmatchedDetections.Add(matchedIndicesD[m]);
                    unmatchedTrackers.Add(matchedIndicesT[m]);
                }
                else
                {
                    matches.Add(new int[] { matchedIndicesD[m], matchedIndicesT[m] });
                }
            }

            // If detection has not been matched, add it to unmatchedDetections
            foreach (var i in detectionIndices.Except(matchedIndicesD))
            {
                unmatchedDetections.Add(i);
            }

            // If tracker has not been matched, add it to unmatchedTrackers
            foreach (var i in trackerIndices.Except(matchedIndicesT))
            {
                unmatchedTrackers.Add(i);
            }

            return (matches, unmatchedDetections, unmatchedTrackers);
        }
    }

    // Implement the KalmanBoxTracker
    public class KalmanBoxTracker
    {
        private static int Count;
        private DiscreteKalmanFilter kf;
        public int TimeSinceUpdate { get; private set; }
        public int Id { get; private set; }
        private List<double[]> history { get; set; }
        private int hits { get; set; }
        public int HitStreak { get; private set; }
        private int age { get; set; }

        public KalmanBoxTracker(double[] bbox)
        {
            // Initialize matrices and vectors
            Matrix<double> stateTransition = Matrix<double>.Build.DenseIdentity(7);
            Matrix<double> control = Matrix<double>.Build.DenseIdentity(7);
            Matrix<double> measurement = Matrix<double>.Build.Dense(4, 7);
            measurement[0, 0] = 1;
            measurement[1, 1] = 1;
            measurement[2, 2] = 1;
            measurement[3, 3] = 1;

            Matrix<double> initialState = Matrix<double>.Build.Dense(1, 7, new double[] 
            {
                bbox[0] + (bbox[2] - bbox[0]) / 2, // x coordinate - center of bounding box
                bbox[1] + (bbox[3] - bbox[1]) / 2, // y coordinate - center of bounding box
                (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]), // Area
                (bbox[2] - bbox[0]) / (bbox[3] - bbox[1]), // Aspect Ratio
                0, // Default initial values for the remaining state variables
                0,
                0
            });

            Matrix<double> initialCovariance = Matrix<double>.Build.DenseIdentity(7) * 10;
            initialCovariance[4, 4] = 1000;
            initialCovariance[5, 5] = 1000;
            initialCovariance[6, 6] = 1000;

            // Instantiate the filter with these matrices and vectors
            this.kf = new DiscreteKalmanFilter(initialState, initialCovariance);

            this.TimeSinceUpdate = 0;
            this.Id = Count;
            this.history = new List<double[]>();
            this.hits = 0;
            this.HitStreak = 0;
            this.age = 0;

            Count += 1;
        }

        public void Update(double[] bbox)
        {
            this.TimeSinceUpdate = 0;
            this.history.Clear();
            this.hits += 1;
            if (this.hits > 0)
            {
                this.HitStreak += 1;
            }

            // Convert bbox to measurement
            Matrix<double> z = Matrix<double>.Build.Dense(4, 1);
            z[0, 0] = bbox[0] + ((bbox[2] - bbox[0]) / 2);
            z[1, 0] = bbox[1] + ((bbox[3] - bbox[1]) / 2);
            z[2, 0] = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]); // Area
            z[3, 0] = (bbox[2] - bbox[0]) / (bbox[3] - bbox[1]); // Aspect Ratio

            // Measurement model
            Matrix<double> H = Matrix<double>.Build.Dense(4, 7);
            H[0, 0] = 1;
            H[1, 1] = 1;
            H[2, 2] = 1;
            H[3, 3] = 1;

            // Covariance of measurements
            Matrix<double> R = Matrix<double>.Build.DenseIdentity(4);

            // Update the Kalman filter
            this.kf.Update(z, H, R);
        }

        public double[] Predict()
        {
            // Define the state transition matrix
            Matrix<double> F = Matrix<double>.Build.DenseIdentity(7);

            // Define the plant noise covariance
            Matrix<double> Q = Matrix<double>.Build.DenseIdentity(7);

            // Perform a prediction using the Kalman filter
            this.kf.Predict(F, Q);

            this.age += 1;
            this.TimeSinceUpdate += 1;

            double x = this.kf.State[0, 0];
            double y = this.kf.State[1, 0];
            double s = this.kf.State[2, 0];
            double r = this.kf.State[3, 0];

            double width = Math.Sqrt(s * r);
            double height = s / width;

            double[] bbox = new double[4];
            bbox[0] = x - width / 2;
            bbox[1] = y - height / 2;
            bbox[2] = x + width / 2;
            bbox[3] = y + height / 2;

            this.history.Add(bbox);
            return bbox;
        }

        public Vector<float> GetState()
        {
            // Implement the state getting function here
            var bbox = ConvertXToBbox(history.Last());
            return bbox;
        }

        // Convert x to bbox
        public static Vector<float> ConvertXToBbox(double[] xState)
        {
            var w = (float)Math.Sqrt(xState[2] * xState[3]);
            var h = (float)(xState[2] / w);
            var x = (float)xState[0] - w / 2;
            var y = (float)xState[1] - h / 2;
            var bbox = Vector<float>.Build.DenseOfArray(new float[] { x, y, x + w, y + h });
            return bbox;
        }
    }
}
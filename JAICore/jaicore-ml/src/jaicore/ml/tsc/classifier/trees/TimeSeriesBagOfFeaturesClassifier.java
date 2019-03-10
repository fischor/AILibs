package jaicore.ml.tsc.classifier.trees;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import jaicore.basic.sets.SetUtil.Pair;
import jaicore.ml.core.exception.PredictionException;
import jaicore.ml.tsc.classifier.ASimplifiedTSClassifier;
import jaicore.ml.tsc.dataset.TimeSeriesDataset;
import jaicore.ml.tsc.features.TimeSeriesFeature;
import jaicore.ml.tsc.util.TimeSeriesUtil;
import jaicore.ml.tsc.util.WekaUtil;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;

// Implementation of the TSBF classifier
public class TimeSeriesBagOfFeaturesClassifier extends ASimplifiedTSClassifier<Integer> {

	/**
	 * Log4j logger.
	 */
	private static final Logger LOGGER = LoggerFactory.getLogger(TimeSeriesBagOfFeaturesClassifier.class);

	private RandomForest subseriesClf;
	private RandomForest finalClf;
	private int numBins = 10;
	private int numClasses;
	private int[][][] intervals;
	private int[][] subseries;

	private boolean useZNormalization = false;

	public TimeSeriesBagOfFeaturesClassifier(final int seed) {
		// Default configuration
		super(new TimeSeriesBagOfFeaturesAlgorithm(seed, 10, 10, 0.1d, 5, false));
		this.useZNormalization = false;
	}

	public TimeSeriesBagOfFeaturesClassifier(final int seed, final int numBins, final int numFolds, final double zProp,
			final int minIntervalLength) {
		super(new TimeSeriesBagOfFeaturesAlgorithm(seed, numBins, numFolds, zProp, minIntervalLength, false));

		this.numBins = numBins;
	}

	public TimeSeriesBagOfFeaturesClassifier(final int seed, final int numBins, final int numFolds, final double zProp,
			final int minIntervalLength, final boolean useZNormalization) {
		super(new TimeSeriesBagOfFeaturesAlgorithm(seed, numBins, numFolds, zProp, minIntervalLength,
				useZNormalization));

		this.numBins = numBins;
		this.useZNormalization = useZNormalization;
	}

	@Override
	public Integer predict(double[] univInstance) throws PredictionException {
		// TODO Auto-generated method stub
		// Z-Normalize if enabled
		if (this.useZNormalization) {
			univInstance = TimeSeriesUtil.zNormalize(univInstance, true);
		}

		// Generate features and interval instances
		double[][] intervalFeatures = new double[intervals.length][(intervals[0].length + 1) * 3 + 2];

		for (int i = 0; i < intervals.length; i++) {
			// Feature generation for each interval
			for (int j = 0; j < intervals[i].length; j++) {
				double[] tmpFeatures = TimeSeriesFeature.getFeatures(univInstance, intervals[i][j][0],
						intervals[i][j][1] - 1, TimeSeriesBagOfFeaturesAlgorithm.USE_BIAS_CORRECTION);

				intervalFeatures[i][j * 3] = tmpFeatures[0];
				intervalFeatures[i][j * 3 + 1] = tmpFeatures[1] * tmpFeatures[1];
				intervalFeatures[i][j * 3 + 2] = tmpFeatures[2];
			}

			// Feature generation for each subseries itself
			double[] subseriesFeatures = TimeSeriesFeature.getFeatures(univInstance, this.subseries[i][0],
					this.subseries[i][1] - 1, TimeSeriesBagOfFeaturesAlgorithm.USE_BIAS_CORRECTION);
			intervalFeatures[i][intervals[i].length * 3] = subseriesFeatures[0];
			intervalFeatures[i][intervals[i].length * 3 + 1] = subseriesFeatures[1] * subseriesFeatures[1];
			intervalFeatures[i][intervals[i].length * 3 + 2] = subseriesFeatures[2];

			// Add start and end indices of subseries to features
			intervalFeatures[i][intervalFeatures[i].length - 2] = this.subseries[i][0];
			intervalFeatures[i][intervalFeatures[i].length - 1] = this.subseries[i][1];
		}

		// Prepare Weka instances for generated features
		ArrayList<double[][]> subseriesValueMatrices = new ArrayList<>();
		subseriesValueMatrices.add(intervalFeatures);
		TimeSeriesDataset subseriesDataset = new TimeSeriesDataset(subseriesValueMatrices);
		Instances subseriesInstances = WekaUtil.simplifiedTimeSeriesDatasetToWekaInstances(subseriesDataset, IntStream
				.rangeClosed(0, this.numClasses - 1).boxed().map(i -> String.valueOf(i)).collect(Collectors.toList()));

		// Predict probabilities using the subseries Random Forest classifier
		double[][] probs = null;
		int[] predictedTargets = new int[subseriesInstances.numInstances()];
		try {
			probs = this.subseriesClf.distributionsForInstances(subseriesInstances);
			for (int i = 0; i < subseriesInstances.numInstances(); i++) {
				predictedTargets[i] = (int) this.subseriesClf.classifyInstance(subseriesInstances.get(i));
			}
		} catch (Exception e) {
			final String errorMessage = "Cannot derive the probabilities using the subseries classifier due to an internal Weka exception.";
			LOGGER.warn(errorMessage, e);
			throw new PredictionException(errorMessage, e);
		}

		// Discretize probabilities and create histograms for final Weka instance
		int[][] discretizedProbs = TimeSeriesBagOfFeaturesAlgorithm.discretizeProbs(this.numBins, probs);
		Pair<int[][][], int[][]> histFreqPair = TimeSeriesBagOfFeaturesAlgorithm
				.formHistogramsAndRelativeFreqs(discretizedProbs, predictedTargets, 1, this.numClasses, this.numBins);
		int[][][] histograms = histFreqPair.getX();
		int[][] relativeFrequencies = histFreqPair.getY();

		// Prepare final Weka instance
		double[][] finalHistogramInstances = TimeSeriesBagOfFeaturesAlgorithm.generateHistogramInstances(histograms,
				relativeFrequencies);
		ArrayList<double[][]> finalMatrices = new ArrayList<>();
		finalMatrices.add(finalHistogramInstances);
		TimeSeriesDataset finalDataset = new TimeSeriesDataset(finalMatrices);
		Instances finalInstances = WekaUtil.simplifiedTimeSeriesDatasetToWekaInstances(finalDataset, IntStream
				.rangeClosed(0, this.numClasses - 1).boxed().map(i -> String.valueOf(i)).collect(Collectors.toList()));

		// Ensure that only on instance has been generated out of the given
		// probabilities
		if (finalInstances.size() != 1) {
			final String errorMessage = "There should be only one instance given to the final Random Forest classifier.";
			throw new PredictionException(errorMessage, new IllegalStateException(errorMessage));
		}

		// Predict using the generated Weka instance
		try {
			int pred = (int) this.finalClf.classifyInstance(finalInstances.firstInstance());
			return pred;
		} catch (Exception e) {
			throw new PredictionException("Could not predict instance due to an internal Weka exception.", e);
		}
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public Integer predict(List<double[]> multivInstance) throws PredictionException {
		throw new UnsupportedOperationException("Multivariate prediction is not supported yet.");
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public List<Integer> predict(TimeSeriesDataset dataset) throws PredictionException {
		// Uses the prediction of single instances
		final List<Integer> result = new ArrayList<>();
		for (int i = 0; i < dataset.getValues(0).length; i++) {
			result.add(this.predict(dataset.getValues(0)[i]));
		}
		return result;
	}

	/**
	 * @return the subseriesClf
	 */
	public RandomForest getSubseriesClf() {
		return subseriesClf;
	}

	/**
	 * @param subseriesClf
	 *            the subseriesClf to set
	 */
	public void setSubseriesClf(RandomForest subseriesClf) {
		this.subseriesClf = subseriesClf;
	}

	/**
	 * @return the finalClf
	 */
	public RandomForest getFinalClf() {
		return finalClf;
	}

	/**
	 * @param finalClf
	 *            the finalClf to set
	 */
	public void setFinalClf(RandomForest finalClf) {
		this.finalClf = finalClf;
	}

	/**
	 * @return the numBins
	 */
	public int getNumBins() {
		return numBins;
	}

	/**
	 * @param numBins
	 *            the numBins to set
	 */
	public void setNumBins(int numBins) {
		this.numBins = numBins;
	}

	/**
	 * @return the numClasses
	 */
	public int getNumClasses() {
		return numClasses;
	}

	/**
	 * @param numClasses
	 *            the numClasses to set
	 */
	public void setNumClasses(int numClasses) {
		this.numClasses = numClasses;
	}

	/**
	 * @return the intervals
	 */
	public int[][][] getIntervals() {
		return intervals;
	}

	/**
	 * @param intervals
	 *            the intervals to set
	 */
	public void setIntervals(int[][][] intervals) {
		this.intervals = intervals;
	}

	/**
	 * @return the subseries
	 */
	public int[][] getSubseries() {
		return subseries;
	}

	/**
	 * @param subseries
	 *            the subseries to set
	 */
	public void setSubseries(int[][] subseries) {
		this.subseries = subseries;
	}

}

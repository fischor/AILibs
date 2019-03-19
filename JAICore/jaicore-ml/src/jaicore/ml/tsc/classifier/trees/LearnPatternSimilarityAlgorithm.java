package jaicore.ml.tsc.classifier.trees;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import jaicore.basic.TimeOut;
import jaicore.basic.algorithm.AlgorithmExecutionCanceledException;
import jaicore.basic.algorithm.IAlgorithm;
import jaicore.basic.algorithm.IAlgorithmConfig;
import jaicore.basic.algorithm.events.AlgorithmEvent;
import jaicore.basic.algorithm.exceptions.AlgorithmException;
import jaicore.ml.core.exception.PredictionException;
import jaicore.ml.tsc.classifier.ASimplifiedTSCAlgorithm;
import jaicore.ml.tsc.dataset.TimeSeriesDataset;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Algorithm training a {@link LearnPatternSimilarityClassifier} as described in
 * Baydogan, Mustafa & Runger, George. (2015). Time series representation and
 * similarity based on local autopatterns. Data Mining and Knowledge Discovery.
 * 30. 1-34. 10.1007/s10618-015-0425-y.
 * 
 * @author Julian Lienen
 *
 */
public class LearnPatternSimilarityAlgorithm
		extends ASimplifiedTSCAlgorithm<Integer, LearnPatternSimilarityClassifier> {

	/**
	 * Log4j logger
	 */
	private static final Logger LOGGER = LoggerFactory.getLogger(LearnPatternSimilarityAlgorithm.class);

	/**
	 * Number of trees being trained.
	 */
	private int numTrees;

	/**
	 * Maximum depth of the trained trees.
	 */
	private int maxTreeDepth;

	/**
	 * Number of segments used for feature generation for each tree.
	 */
	private int numSegments;

	/**
	 * Seed used for randomized operations.
	 */
	private int seed;

	/**
	 * See {@link IAlgorithm#getTimeout()}.
	 */
	private TimeOut timeout = new TimeOut(Integer.MAX_VALUE, TimeUnit.SECONDS);

	/**
	 * Standard constructor.
	 * 
	 * @param seed
	 *            See {@link LearnPatternSimilarityAlgorithm#seed}.
	 * @param numTrees
	 *            See {@link LearnPatternSimilarityAlgorithm#numTrees}.
	 * @param maxTreeDepth
	 *            See {@link LearnPatternSimilarityAlgorithm#maxTreeDepth}.
	 * @param numSegments
	 *            See {@link LearnPatternSimilarityAlgorithm#numSegments}.
	 */
	public LearnPatternSimilarityAlgorithm(final int seed, final int numTrees, final int maxTreeDepth,
			final int numSegments) {
		super();
		this.seed = seed;
		this.numTrees = numTrees;
		this.maxTreeDepth = maxTreeDepth;
		this.numSegments = numSegments;
	}

	@Override
	public LearnPatternSimilarityClassifier call()
			throws InterruptedException, AlgorithmExecutionCanceledException, TimeoutException, AlgorithmException {
		// Training procedure
		long beginTimeMs = System.currentTimeMillis();

		TimeSeriesDataset data = this.getInput();
		if (data == null || data.isEmpty())
			throw new IllegalStateException("The time series input data must not be null or empty!");

		final double[][] dataMatrix = data.getValuesOrNull(0);
		if (dataMatrix == null)
			throw new IllegalArgumentException(
					"Value matrix must be a valid 2D matrix containing the time series values for all instances!");
		final int[] targetMatrix = data.getTargets();

		final int timeSeriesLength = dataMatrix[0].length;

		int minLength = (int) (0.1d * timeSeriesLength);
		int maxLength = (int) (0.9d * timeSeriesLength);
		
		Random random = new Random(this.seed);
		final int[][] segments = new int[this.numTrees][this.numSegments]; // Refers to matrix A in tsc algorithm
																			// description
		final int[][] segmentsDifference = new int[this.numTrees][this.numSegments]; // Refers to matrix B in tsc
																						// algorithm description

		final int[] lengthPerTree = new int[this.numTrees];
		final int[] classAttIndex = new int[this.numTrees];

		final RandomRegressionTree[] trees = new RandomRegressionTree[this.numTrees];
		final int[] numLeavesPerTree = new int[this.numTrees];
		final int[][][] leafNodeCounts = new int[data.getNumberOfInstances()][this.numTrees][];

		ArrayList<Attribute> attributes = new ArrayList<>();
		for (int j = 0; j < 2 * this.numSegments; j++) {
			attributes.add(new Attribute("val" + j));
		}

		for (int i = 0; i < numTrees; i++) {
			if ((System.currentTimeMillis() - beginTimeMs) > this.getTimeout().milliseconds()) {
				throw new TimeoutException("Timeout in tree iteration " + i + ".");
			}

			// Generate subseries length
			lengthPerTree[i] = random.nextInt(maxLength - minLength) + minLength;

			// Generate random subseries locations as described in chapter 3.1 and random
			// subseries difference locations as described in chapter 3.4
			this.generateSegmentsAndDifferencesForTree(segments[i], segmentsDifference[i], lengthPerTree[i],
					timeSeriesLength, random);

			// Generate subseries features
			Instances seqInstances = generateSubseriesFeaturesInstances(attributes, lengthPerTree[i], segments[i],
					segmentsDifference[i], dataMatrix);

			classAttIndex[i] = random.nextInt(attributes.size());
			seqInstances.setClassIndex(classAttIndex[i]);

			trees[i] = this.initializeRegressionTree(seqInstances.numInstances());

			try {
				trees[i].buildClassifier(seqInstances);
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
				throw new AlgorithmException("Could not build tree in iteration " + i
						+ " due to the following exception: " + e.getMessage());
			}

			numLeavesPerTree[i] = trees[i].nosLeafNodes;
			for (int inst = 0; inst < data.getNumberOfInstances(); inst++) {
				leafNodeCounts[inst][i] = new int[numLeavesPerTree[i]];

				for (int len = 0; len < lengthPerTree[i]; len++) {
					int instanceIdx = inst * lengthPerTree[i] + len;

					try {
						collectLeafCounts(leafNodeCounts[inst][i], seqInstances.get(instanceIdx), trees[i]);
					} catch (PredictionException e1) {
						// TODO Auto-generated catch block
						e1.printStackTrace();
					}
				}
			}
		}

		// Update model
		this.model.setSegments(segments);
		this.model.setSegmentsDifference(segmentsDifference);
		this.model.setLengthPerTree(lengthPerTree);
		this.model.setClassAttIndexPerTree(classAttIndex);
		this.model.setTrees(trees);
		this.model.setTrainLeafNodes(leafNodeCounts);
		this.model.setTrainTargets(targetMatrix);
		this.model.setAttributes(attributes);

		this.model.setTrained(true);

		return this.model;
	}

	public void generateSegmentsAndDifferencesForTree(final int[] segments, final int[] segmentsDifference,
			final int length, final int timeSeriesLength, final Random random) {
		for (int i = 0; i < this.numSegments; i++) {
			segments[i] = random.nextInt(timeSeriesLength - length); // Length is always l
			segmentsDifference[i] = random.nextInt(timeSeriesLength - length - 1);
		}
	}

	public RandomRegressionTree initializeRegressionTree(final int numInstances) {
		RandomRegressionTree regTree = new RandomRegressionTree();
		regTree.setSeed(this.seed);
		regTree.setMaxDepth(this.maxTreeDepth);
		regTree.setKValue(1);
		// regTree.setMinVarianceProp(1e-5);
		regTree.setMinNum((int) (numInstances * 0.01)); // TODO
		return regTree;
	}

	public static void collectLeafCounts(final int[] leafNodeCountsForInstance, final Instance instance,
			final RandomRegressionTree regTree) throws PredictionException {
		try {
			regTree.distributionForInstance(instance);
		} catch (Exception e) {
			throw new PredictionException("Could not predict the distribution for instance for the given instance '"
					+ instance.toString() + "' due to an internal Weka exception.", e);
		}
		int leafNodeIdx = RandomRegressionTree.lastNode;
		leafNodeCountsForInstance[leafNodeIdx]++;
	}

	public static Instances generateSubseriesFeaturesInstances(final ArrayList<Attribute> attributes, final int length,
			final int[] segments, final int[] segmentsDifference, final double[][] dataMatrix) {
		Instances seqInstances = new Instances("SeqFeatures", attributes, dataMatrix.length * length);
		for (int inst = 0; inst < dataMatrix.length; inst++) {
			double[] instValues = dataMatrix[inst];
			for (int len = 0; len < length; len++) {
				seqInstances.add(
						generateSubseriesFeatureInstance(instValues, segments, segmentsDifference, len));
			}
		}
		return seqInstances;
	}

	public static Instance generateSubseriesFeatureInstance(final double[] instValues, final int[] segments,
			final int[] segmentsDifference, final int len) {
		if (segments.length != segmentsDifference.length)
			throw new IllegalArgumentException(
					"The number of segments and the number of segments differences must be the same!");
		if (instValues.length < len)
			throw new IllegalArgumentException("If the segments' length is set to '" + len
					+ "', the number of time series variables must be greater or equals!");

		DenseInstance instance = new DenseInstance(2 * segments.length);
		for (int seq = 0; seq < segments.length; seq++) {
			instance.setValue(seq * 2, instValues[segments[seq] + len]);

			double difference = instValues[segmentsDifference[seq] + len + 1]
					- instValues[segmentsDifference[seq] + len];
			instance.setValue(seq * 2 + 1, difference);
		}
		return instance;
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public int getNumCPUs() {
		LOGGER.warn(
				"Multithreading is not supported for LearnPatternSimilarity yet. Therefore, the number of CPUs is not considered.");
		return 1;
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public void setNumCPUs(int numberOfCPUs) {
		LOGGER.warn(
				"Multithreading is not supported for LearnShapelets yet. Therefore, the number of CPUs is not considered.");
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public void setTimeout(long timeout, TimeUnit timeUnit) {
		this.timeout = new TimeOut(timeout, timeUnit);
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public void setTimeout(TimeOut timeout) {
		this.timeout = timeout;
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public TimeOut getTimeout() {
		return this.timeout;
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public void registerListener(Object listener) {
		throw new UnsupportedOperationException("The operation to be performed is not supported.");
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public AlgorithmEvent nextWithException() {
		throw new UnsupportedOperationException("The operation to be performed is not supported.");
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public Iterator<AlgorithmEvent> iterator() {
		throw new UnsupportedOperationException("The operation to be performed is not supported.");
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public boolean hasNext() {
		throw new UnsupportedOperationException("The operation to be performed is not supported.");
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public AlgorithmEvent next() {
		throw new UnsupportedOperationException("The operation to be performed is not supported.");
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public void cancel() {
		throw new UnsupportedOperationException("The operation to be performed is not supported.");
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public IAlgorithmConfig getConfig() {
		throw new UnsupportedOperationException("The operation to be performed is not supported.");
	}

	/**
	 * @return the numTrees
	 */
	public int getNumTrees() {
		return numTrees;
	}

	/**
	 * @param numTrees
	 *            the numTrees to set
	 */
	public void setNumTrees(int numTrees) {
		this.numTrees = numTrees;
	}

	/**
	 * @return the maxTreeDepth
	 */
	public int getMaxTreeDepth() {
		return maxTreeDepth;
	}

	/**
	 * @param maxTreeDepth
	 *            the maxTreeDepth to set
	 */
	public void setMaxTreeDepth(int maxTreeDepth) {
		this.maxTreeDepth = maxTreeDepth;
	}

	/**
	 * @return the numSegments
	 */
	public int getNumSegments() {
		return numSegments;
	}

	/**
	 * @param numSegments
	 *            the numSegments to set
	 */
	public void setNumSegments(int numSegments) {
		this.numSegments = numSegments;
	}

	/**
	 * @return the seed
	 */
	public int getSeed() {
		return seed;
	}

	/**
	 * @param seed
	 *            the seed to set
	 */
	public void setSeed(int seed) {
		this.seed = seed;
	}

}

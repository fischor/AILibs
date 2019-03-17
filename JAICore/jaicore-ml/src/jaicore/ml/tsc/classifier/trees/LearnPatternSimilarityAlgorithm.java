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
import jaicore.basic.algorithm.IAlgorithmConfig;
import jaicore.basic.algorithm.events.AlgorithmEvent;
import jaicore.basic.algorithm.exceptions.AlgorithmException;
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

	private int numTrees;
	private int maxTreeDepth;
	private int numSegments;
	private int seed;

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
		// TODO Auto-generated method stub
		// Training procedure
		long beginTime = System.currentTimeMillis();

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
			// Generate subseries length
			lengthPerTree[i] = random.nextInt(maxLength - minLength) + minLength;

			// Generate random subseries locations as described in chapter 3.1 and random
			// subseries difference locations as described in chapter 3.4
			for (int j = 0; j < this.numSegments; j++) {
				segments[i][j] = random.nextInt(timeSeriesLength - lengthPerTree[i]); // Length is always l
				segmentsDifference[i][j] = random.nextInt(timeSeriesLength - lengthPerTree[i] - 1);
			}

			// Generate subseries features
			Instances seqInstances = generateSubseriesFeaturesInstances(attributes, lengthPerTree[i], this.numSegments,
					segments[i], segmentsDifference[i], data.getNumberOfInstances(), dataMatrix);

			classAttIndex[i] = random.nextInt(attributes.size());
			seqInstances.setClassIndex(classAttIndex[i]);

			trees[i] = new RandomRegressionTree();
			trees[i].setSeed(this.seed);
			trees[i].setMaxDepth(this.maxTreeDepth);
			trees[i].setKValue(1);
			// trees[i].setBreakTiesRandomly(true);
			trees[i].setMinVarianceProp(1e-5);
			trees[i].setMinNum((int) (seqInstances.numInstances() * 0.01));

			try {
				trees[i].buildClassifier(seqInstances);
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
				throw new AlgorithmException("Could not build tree in iteration " + i + ".");
			}

			numLeavesPerTree[i] = trees[i].nosLeafNodes;
			for (int inst = 0; inst < data.getNumberOfInstances(); inst++) {
				leafNodeCounts[inst][i] = new int[numLeavesPerTree[i]];

				for (int len = 0; len < lengthPerTree[i]; len++) {
					int instanceIdx = inst * lengthPerTree[i] + len;
					try {
						trees[i].distributionForInstance(seqInstances.get(instanceIdx));
					} catch (Exception e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
						throw new AlgorithmException("");
					}
					int leafNodeIdx = RandomRegressionTree.lastNode;
					leafNodeCounts[inst][i][leafNodeIdx]++;
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

	public static Instances generateSubseriesFeaturesInstances(final ArrayList<Attribute> attributes, final int length,
			final int numSegments, final int[] segments, final int[] segmentsDifference,
			final int numInstances, final double[][] dataMatrix) {
		Instances seqInstances = new Instances("SeqFeatures", attributes, numInstances * length);
		for (int inst = 0; inst < numInstances; inst++) {
			double[] instValues = dataMatrix[inst];
			for (int len = 0; len < length; len++) {
				seqInstances.add(
						generateSubseriesFeatureInstance(instValues, numSegments, segments, segmentsDifference, len));
			}
		}
		return seqInstances;
	}

	public static Instance generateSubseriesFeatureInstance(final double[] instValues, final int numSegments,
			final int[] segments, final int[] segmentsDifference, final int len) {
		DenseInstance instance = new DenseInstance(2 * numSegments);
		for (int seq = 0; seq < numSegments; seq++) {
			instance.setValue(seq * 2, instValues[segments[seq] + len]);

			double difference = instValues[segmentsDifference[seq] + len]
					- instValues[segmentsDifference[seq] + len + 1];
			instance.setValue(seq * 2 + 1, difference);
		}
		return instance;
	}

	@Override
	public int getNumCPUs() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public void setNumCPUs(int numberOfCPUs) {
		// TODO Auto-generated method stub

	}

	@Override
	public void setTimeout(long timeout, TimeUnit timeUnit) {
		// TODO Auto-generated method stub

	}

	@Override
	public void setTimeout(TimeOut timeout) {
		// TODO Auto-generated method stub

	}

	@Override
	public TimeOut getTimeout() {
		// TODO Auto-generated method stub
		return null;
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
}

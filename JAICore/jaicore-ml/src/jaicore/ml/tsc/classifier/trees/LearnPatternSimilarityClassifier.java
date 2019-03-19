package jaicore.ml.tsc.classifier.trees;

import java.util.ArrayList;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import jaicore.ml.core.exception.PredictionException;
import jaicore.ml.tsc.classifier.ASimplifiedTSClassifier;
import jaicore.ml.tsc.dataset.TimeSeriesDataset;
import jaicore.ml.tsc.util.MathUtil;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Class representing the Learn Pattern Similarity classifier as described in
 * Baydogan, Mustafa & Runger, George. (2015). Time series representation and
 * similarity based on local autopatterns. Data Mining and Knowledge Discovery.
 * 30. 1-34. 10.1007/s10618-015-0425-y.
 * 
 * This classifier currently only supports univariate time series prediction.
 * 
 * @author Julian Lienen
 *
 */
public class LearnPatternSimilarityClassifier extends ASimplifiedTSClassifier<Integer> {

	/**
	 * Log4j logger
	 */
	private static final Logger LOGGER = LoggerFactory.getLogger(LearnPatternSimilarityClassifier.class);

	// Hyperparameters
	private int numTrees;
	private int numSegments;

	// Trained parameters
	private int[][] segments;
	private int[][] segmentsDifference;
	private int[] lengthPerTree;
	private int[] classAttIndexPerTree;
	private RandomRegressionTree[] trees;

	private int[][][] trainLeafNodes;
	private int[] trainTargets;

	private ArrayList<Attribute> attributes;

	public LearnPatternSimilarityClassifier(final int seed, final int numTrees, final int maxTreeDepth, final int numSegments) {
		super(new LearnPatternSimilarityAlgorithm(seed, numTrees, maxTreeDepth, numSegments));
		this.numTrees = numTrees;
		this.numSegments = numSegments;
	}

	@Override
	public Integer predict(double[] univInstance) throws PredictionException {
		if (!this.isTrained())
			throw new PredictionException("Model has not been built before!");

		if (univInstance == null)
			throw new IllegalArgumentException("Instance to be predicted must not be null or empty!");
		
		int[][] leafNodeCounts = new int[this.numTrees][];

		for (int i = 0; i < this.numTrees; i++) {

			// Generate subseries features
			Instances seqInstances = new Instances("SeqFeatures", attributes, lengthPerTree[i]);

			for (int len = 0; len < lengthPerTree[i]; len++) {
				Instance instance = LearnPatternSimilarityAlgorithm.generateSubseriesFeatureInstance(univInstance,
						segments[i], segmentsDifference[i], len);
				seqInstances.add(instance);
			}

			seqInstances.setClassIndex(classAttIndexPerTree[i]);
			leafNodeCounts[i] = new int[trees[i].nosLeafNodes];
			
			for(int inst = 0; inst< seqInstances.numInstances(); inst++) {
				LearnPatternSimilarityAlgorithm.collectLeafCounts(leafNodeCounts[i], seqInstances.get(inst), trees[i]);
			}
		}
		return trainTargets[findNearestInstanceIndex(leafNodeCounts)];
	}

	/**
	 * Performs a simple nearest neighbor search on the stored
	 * <code>trainLeafNodes</code> for the given <code>leafNodeCounts</code> using
	 * Manhattan distance.
	 * 
	 * @param leafNodeCounts
	 * @return
	 */
	public int findNearestInstanceIndex(final int[][] leafNodeCounts) {
		double minDistance = Double.MAX_VALUE;
		int nearestInstIdx = 0;
		for (int inst = 0; inst < this.trainLeafNodes.length; inst++) {
			double tmpDist = 0;
			for (int i = 0; i < trainLeafNodes[inst].length; i++) {
				tmpDist += MathUtil.intManhattanDistance(trainLeafNodes[inst][i], leafNodeCounts[i]);
			}

			if (tmpDist < minDistance) {
				minDistance = tmpDist;
				nearestInstIdx = inst;
			}
		}
		return nearestInstIdx;
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public Integer predict(List<double[]> multivInstance) throws PredictionException {
		LOGGER.warn(
				"Dataset to be predicted is multivariate but only first time series (univariate) will be considered.");

		return predict(multivInstance.get(0));
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public List<Integer> predict(TimeSeriesDataset dataset) throws PredictionException {
		if (!this.isTrained())
			throw new PredictionException("Model has not been built before!");

		if (dataset.isMultivariate())
			throw new UnsupportedOperationException("Multivariate instances are not supported yet.");

		if (dataset == null || dataset.isEmpty())
			throw new IllegalArgumentException("Dataset to be predicted must not be null or empty!");

		double[][] data = dataset.getValuesOrNull(0);
		List<Integer> predictions = new ArrayList<>();
		LOGGER.debug("Starting prediction...");
		for (int i = 0; i < data.length; i++) {
			predictions.add(this.predict(data[i]));
		}
		LOGGER.debug("Finished prediction.");
		return predictions;
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
	 * @return the segments
	 */
	public int[][] getSegments() {
		return segments;
	}

	/**
	 * @param segments
	 *            the segments to set
	 */
	public void setSegments(int[][] segments) {
		this.segments = segments;
	}

	/**
	 * @return the segmentsDifference
	 */
	public int[][] getSegmentsDifference() {
		return segmentsDifference;
	}

	/**
	 * @param segmentsDifference
	 *            the segmentsDifference to set
	 */
	public void setSegmentsDifference(int[][] segmentsDifference) {
		this.segmentsDifference = segmentsDifference;
	}

	/**
	 * @return the lengthPerTree
	 */
	public int[] getLengthPerTree() {
		return lengthPerTree;
	}

	/**
	 * @param lengthPerTree
	 *            the lengthPerTree to set
	 */
	public void setLengthPerTree(int[] lengthPerTree) {
		this.lengthPerTree = lengthPerTree;
	}

	/**
	 * @return the classAttIndexPerTree
	 */
	public int[] getClassAttIndexPerTree() {
		return classAttIndexPerTree;
	}

	/**
	 * @param classAttIndexPerTree
	 *            the classAttIndexPerTree to set
	 */
	public void setClassAttIndexPerTree(int[] classAttIndexPerTree) {
		this.classAttIndexPerTree = classAttIndexPerTree;
	}

	/**
	 * @return the trees
	 */
	public RandomRegressionTree[] getTrees() {
		return trees;
	}

	/**
	 * @param trees
	 *            the trees to set
	 */
	public void setTrees(RandomRegressionTree[] trees) {
		this.trees = trees;
	}

	/**
	 * @return the trainLeafNodes
	 */
	public int[][][] getTrainLeafNodes() {
		return trainLeafNodes;
	}

	/**
	 * @param trainLeafNodes
	 *            the trainLeafNodes to set
	 */
	public void setTrainLeafNodes(int[][][] trainLeafNodes) {
		this.trainLeafNodes = trainLeafNodes;
	}

	/**
	 * @return the trainTargets
	 */
	public int[] getTrainTargets() {
		return trainTargets;
	}

	/**
	 * @param trainTargets
	 *            the trainTargets to set
	 */
	public void setTrainTargets(int[] trainTargets) {
		this.trainTargets = trainTargets;
	}

	/**
	 * @return the attributes
	 */
	public ArrayList<Attribute> getAttributes() {
		return attributes;
	}

	/**
	 * @param attributes
	 *            the attributes to set
	 */
	public void setAttributes(ArrayList<Attribute> attributes) {
		this.attributes = attributes;
	}
}

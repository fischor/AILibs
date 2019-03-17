package jaicore.ml.tsc.classifier;

import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import jaicore.basic.TimeOut;
import jaicore.basic.algorithm.IAlgorithm;
import jaicore.basic.algorithm.IAlgorithmConfig;
import jaicore.basic.algorithm.events.AlgorithmEvent;
import jaicore.ml.tsc.dataset.TimeSeriesDataset;
import jaicore.ml.tsc.util.MathUtil;
import jaicore.ml.tsc.util.TimeSeriesUtil;
import jaicore.ml.tsc.util.WekaUtil;
import weka.clusterers.SimpleKMeans;
import weka.core.Debug.Random;
import weka.core.Instances;

/**
 * Generalized Shapelets Learning implementation for
 * <code>LearnShapeletsClassifier</code> published in "J. Grabocka, N.
 * Schilling, M. Wistuba, L. Schmidt-Thieme: Learning Time-Series Shapelets"
 * (https://www.ismll.uni-hildesheim.de/pub/pdfs/grabocka2014e-kdd.pdf)
 * 
 * @author Julian Lienen
 *
 */
public class LearnShapeletsAlgorithm extends ASimplifiedTSCAlgorithm<Integer, LearnShapeletsClassifier> {

	/**
	 * The log4j logger.
	 */
	private static final Logger LOGGER = LoggerFactory.getLogger(LearnShapeletsAlgorithm.class);

	/**
	 * Parameter which determines how many of the most-informative shapelets should
	 * be used.
	 */
	private int K;
	/**
	 * The learning rate used within the SGD.
	 */
	private double learningRate;
	/**
	 * The regularization used wihtin the SGD.
	 */
	private double regularization;
	/**
	 * The number of scales used for the shapelet lengths.
	 */
	private int scaleR;
	/**
	 * The minimum shapelet of the shapelets to be learned. Internally derived by
	 * the time series lengths and the <code>minShapeLengthPercentage</code>.
	 */
	private int minShapeLength;
	/**
	 * The minimum shape length percentage used to calculate the minimum shape
	 * length.
	 */
	private double minShapeLengthPercentage;
	/**
	 * The maximum iterations used for the SGD.
	 */
	private int maxIter;
	/**
	 * The seed used within the initialization and SGD.
	 */
	private int seed;

	/**
	 * The number of instances.
	 */
	private int I;
	/**
	 * The number of attributes (i. e. the time series lengths without the class
	 * attribute).
	 */
	private int Q;
	/**
	 * The number of classes.
	 */
	private int C;

	/**
	 * Indicator whether Bessel's correction should be used when normalizing arrays.
	 */
	public static final boolean USE_BIAS_CORRECTION = true;

	/**
	 * Predefined alpha parameter used within the calculations.
	 */
	public static double ALPHA = -30d; // Used in implementation. Paper says -100d

	/**
	 * Epsilon value used to prevent dividing by zero occurrences.
	 */
	private static double EPS = 0.000000000000000000001d;

	/**
	 * See {@link IAlgorithm#getTimeout()}.
	 */
	private TimeOut timeout = new TimeOut(Integer.MAX_VALUE, TimeUnit.SECONDS);

	/**
	 * Parameter indicator whether estimation of K (number of learned shapelets)
	 * should be derived from the number of total segments. False by default.
	 */
	private boolean estimateK = false;

	/**
	 * Constructor of the algorithm to train a {@link LearnShapeletsClassifier}.
	 * 
	 * @param K
	 *            See {@link LearnShapeletsAlgorithm#K}
	 * @param learningRate
	 *            See {@link LearnShapeletsAlgorithm#learningRate}
	 * @param regularization
	 *            See {@link LearnShapeletsAlgorithm#regularization}
	 * @param scaleR
	 *            See {@link LearnShapeletsAlgorithm#scaleR}
	 * @param minShapeLengthPercentage
	 *            See {@link LearnShapeletsAlgorithm#minShapeLengthPercentage}
	 * @param maxIter
	 *            See {@link LearnShapeletsAlgorithm#maxIter}
	 * @param seed
	 *            See {@link LearnShapeletsAlgorithm#seed}
	 * @param timeout
	 *            See {@link LearnShapeletsAlgorithm#timeout}
	 */
	public LearnShapeletsAlgorithm(final int K, final double learningRate, final double regularization,
			final int scaleR, final double minShapeLengthPercentage, final int maxIter, final int seed,
			final TimeOut timeout) {
		this.K = K;
		this.learningRate = learningRate;
		this.regularization = regularization;
		this.scaleR = scaleR;
		this.maxIter = maxIter;
		this.seed = seed;
		this.minShapeLengthPercentage = minShapeLengthPercentage;
		this.timeout = timeout;
	}

	/**
	 * Constructor of the algorithm to train a {@link LearnShapeletsClassifier}.
	 * 
	 * @param K
	 *            See {@link LearnShapeletsAlgorithm#K}
	 * @param learningRate
	 *            See {@link LearnShapeletsAlgorithm#learningRate}
	 * @param regularization
	 *            See {@link LearnShapeletsAlgorithm#regularization}
	 * @param scaleR
	 *            See {@link LearnShapeletsAlgorithm#scaleR}
	 * @param minShapeLengthPercentage
	 *            See {@link LearnShapeletsAlgorithm#minShapeLengthPercentage}
	 * @param maxIter
	 *            See {@link LearnShapeletsAlgorithm#maxIter}
	 * @param seed
	 *            See {@link LearnShapeletsAlgorithm#seed}
	 */
	public LearnShapeletsAlgorithm(final int K, final double learningRate, final double regularization,
			final int scaleR, final double minShapeLengthPercentage, final int maxIter, final int seed) {
		this.K = K;
		this.learningRate = learningRate;
		this.regularization = regularization;
		this.scaleR = scaleR;
		this.maxIter = maxIter;
		this.seed = seed;
		this.minShapeLengthPercentage = minShapeLengthPercentage;
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
	 * Initializes the tensor <code>S</code> storing the shapelets for each scale.
	 * The initialization is done by deriving inital shapelets from all normalized
	 * segments.
	 * 
	 * @param trainingMatrix
	 *            The training matrix used for the initialization of <code>S</code>.
	 * @return Return the initialized tensor storing an initial guess for the
	 *         shapelets based on the clustering
	 */
	public double[][][] initializeS(final double[][] trainingMatrix) {
		LOGGER.debug("Initializing S...");

		final double[][][] result = new double[this.scaleR][][];

		for (int r = 0; r < this.scaleR; r++) {
			final int numberOfSegments = getNumberOfSegments(this.Q, this.minShapeLength, r);

			final int L = (r + 1) * this.minShapeLength;

			final double[][] tmpSegments = new double[trainingMatrix.length * numberOfSegments][L];

			// Prepare training data for finding the centroids
			for (int i = 0; i < trainingMatrix.length; i++) {
				for (int j = 0; j < numberOfSegments; j++) {
					for (int l = 0; l < L; l++) {
						tmpSegments[i * numberOfSegments + j][l] = trainingMatrix[i][j + l];
					}
					tmpSegments[i * numberOfSegments + j] = TimeSeriesUtil
							.zNormalize(tmpSegments[i * numberOfSegments + j], USE_BIAS_CORRECTION);
				}
			}

			// Transform instances
			Instances wekaInstances = WekaUtil.matrixToWekaInstances(tmpSegments);

			// Cluster using k-Means
			SimpleKMeans kMeans = new SimpleKMeans();
			try {
				kMeans.setNumClusters(this.K);
				kMeans.setSeed(this.seed);
				kMeans.setMaxIterations(100);
				kMeans.buildClusterer(wekaInstances);
			} catch (Exception e) {
				LOGGER.warn(
						"Could not initialize matrix S using kMeans clustering for r={} due to the following problem: {}. "
								+ "Using zero matrix instead (possibly leading to a poor training performance).",
						r, e.getMessage());
				result[r] = new double[this.K][r * this.minShapeLength];
				continue;
			}
			Instances clusterCentroids = kMeans.getClusterCentroids();

			double[][] tmpResult = new double[clusterCentroids.numInstances()][clusterCentroids.numAttributes()];
			for (int i = 0; i < tmpResult.length; i++) {
				double[] instValues = clusterCentroids.get(i).toDoubleArray();
				for (int j = 0; j < tmpResult[i].length; j++) {
					tmpResult[i][j] = instValues[j];
				}
			}
			result[r] = tmpResult;
		}

		LOGGER.debug("Initialized S.");

		return result;
	}

	/**
	 * Main function to train a <code>LearnShapeletsClassifier</code>.
	 */
	@Override
	public LearnShapeletsClassifier call() {
		// Training
		long beginTime = System.currentTimeMillis();

		TimeSeriesDataset data = this.getInput();

		if (data.isMultivariate())
			throw new UnsupportedOperationException("Multivariate datasets are not supported.");
		if (data.isEmpty())
			throw new IllegalArgumentException("The training dataset must not be null!");

		final double[][] dataMatrix = data.getValuesOrNull(0);
		if (dataMatrix == null)
			throw new IllegalArgumentException(
					"Timestamp matrix must be a valid 2D matrix containing the time series values for all instances!");

		// Get occurring classes which can be used for index extraction
		final int[] targetMatrix = data.getTargets();
		final List<Integer> occuringClasses = TimeSeriesUtil.getClassesInDataset(data);

		this.I = (int) data.getNumberOfInstances(); // I
		this.Q = dataMatrix[0].length; // Q
		this.C = occuringClasses.size(); // C

		this.minShapeLength = (int) (this.minShapeLengthPercentage * (double) this.Q);

		// Prepare binary classes
		int[][] Y = new int[this.I][this.C];
		for (int i = 0; i < this.I; i++) {
			Integer instanceClass = targetMatrix[i];
			Y[i][occuringClasses.indexOf(instanceClass)] = 1;
		}

		// Estimate parameter K by the maximum number of segments
		if (this.estimateK) {
			int totalSegments = 0;
			for (int r = 0; r < this.scaleR; r++) {
				final int numberOfSegments = getNumberOfSegments(this.Q, this.minShapeLength, r);
				totalSegments += numberOfSegments * this.I;
			}

			this.K = (int) (Math.log(totalSegments) * (this.C - 1));
		}

		LOGGER.info("Parameters: k={}, learningRate={}, reg={}, r={}, minShapeLength={}, maxIter={}, Q={}, C={}", K,
				learningRate, regularization, scaleR, minShapeLength, maxIter, Q, C);

		// Initialization
		double[][][] S = initializeS(dataMatrix);
		double[][][] S_hist = new double[this.scaleR][][];
		for (int r = 0; r < this.scaleR; r++) {
			S_hist[r] = new double[S[r].length][S[r][0].length];
		}
		double[][][][] D = new double[this.scaleR][][][];
		double[][][][] Xi = new double[this.scaleR][][][];
		double[][][][] Phi = new double[this.scaleR][][][];

		int[] numberOfSegments = new int[this.scaleR];

		for (int r = 0; r < this.scaleR; r++) {
			numberOfSegments[r] = getNumberOfSegments(this.Q, this.minShapeLength, r);
			D[r] = new double[this.I][this.K][numberOfSegments[r]];
			Xi[r] = new double[this.I][this.K][numberOfSegments[r]];
			Phi[r] = new double[this.I][this.K][numberOfSegments[r]];
		}

		Random rand = new Random(this.seed);

		// Initializes the given weights nearly around zeros (as opposed to the paper
		// due to vanish effects)
		double[][][] W = new double[this.C][this.scaleR][this.K];
		double[][][] W_hist = new double[this.C][this.scaleR][this.K];
		double[] W_0 = new double[this.C];
		double[] W_0_hist = new double[this.C];
		for (int i = 0; i < this.C; i++) {
			W_0[i] = 2 * EPS * rand.nextDouble() - 1;
			for (int j = 0; j < this.scaleR; j++) {
				for (int k = 0; k < this.K; k++) {
					W[i][j][k] = 2 * EPS * rand.nextDouble() - 1;
				}
			}
		}

		double[][][] Psi = new double[this.scaleR][this.I][this.K];
		double[][][] M_hat = new double[this.scaleR][this.I][this.K];
		double[][] Theta = new double[this.I][this.C];

		List<Integer> indices = IntStream.range(0, this.I).boxed().collect(Collectors.toList());

		// Stochastic gradient descent
		LOGGER.debug("Starting training for {} iterations...", this.maxIter);
		for (int it = 0; it < this.maxIter; it++) {
			// Shuffle instances
			Collections.shuffle(indices, new Random(this.seed + it));

			for (int idx = 0; idx < this.I; idx++) {
				int i = indices.get(idx);

				// Pre-compute terms
				for (int r = 0; r < this.scaleR; r++) {

					long kBound = S[r].length;
					for (int k = 0; k < kBound; k++) { // this.K

						int J_r = numberOfSegments[r];

						for (int j = 0; j < J_r; j++) {

							double newDValue = calculateD(S, minShapeLength, r, dataMatrix[i], k, j);
							D[r][i][k][j] = newDValue;
							newDValue = Math.exp(ALPHA * newDValue);
							Xi[r][i][k][j] = newDValue;

						}

						double newPsiValue = 0;
						double newMHatValue = 0;

						for (int j = 0; j < J_r; j++) {
							newPsiValue += Xi[r][i][k][j];
							newMHatValue += D[r][i][k][j] * Xi[r][i][k][j];
						}
						Psi[r][i][k] = newPsiValue;

						newMHatValue /= Psi[r][i][k];

						M_hat[r][i][k] = newMHatValue;
					}
				}

				for (int c = 0; c < this.C; c++) {
					double newThetaValue = 0;
					for (int r = 0; r < this.scaleR; r++) {
						for (int k = 0; k < this.K; k++) {

							newThetaValue += M_hat[r][i][k] * W[c][r][k];
						}
					}
					Theta[i][c] = Y[i][c] - MathUtil.sigmoid(newThetaValue);
				}

				// Learn shapelets and classification weights
				for (int c = 0; c < this.C; c++) {
					double gradW_0 = Theta[i][c];

					for (int r = 0; r < this.scaleR; r++) {
						for (int k = 0; k < S[r].length; k++) { // this differs from paper: this.K instead of
																// shapelet length
							double wStep = (-1d) * Theta[i][c] * M_hat[r][i][k]
									+ 2d * this.regularization / (this.I) * W[c][r][k];

							W_hist[c][r][k] += wStep * wStep;

							W[c][r][k] -= (this.learningRate * wStep / Math.sqrt(W_hist[c][r][k] + EPS));

							int J_r = numberOfSegments[r];

							double phiDenominator = 1d / ((r + 1d) * this.minShapeLength * Psi[r][i][k]);

							double[] distDiff = new double[J_r];
							for (int j = 0; j < J_r; j++) {
								distDiff[j] = Xi[r][i][k][j] * (1d + ALPHA * (D[r][i][k][j] - M_hat[r][i][k]));
							}

							for (int l = 0; l < (r + 1) * this.minShapeLength; l++) {
								double shapeletDiff = 0;
								for (int j = 0; j < J_r; j++)
									shapeletDiff += distDiff[j] * (S[r][k][l] - dataMatrix[i][j + l]);

								double sStep = (-1d) * gradW_0 * shapeletDiff * W[c][r][k] * phiDenominator;
								S_hist[r][k][l] += sStep * sStep;

								S[r][k][l] -= this.learningRate * sStep / Math.sqrt(S_hist[r][k][l] + EPS);
							}
						}
					}

					W_0_hist[c] += gradW_0 * gradW_0;
					W_0[c] += this.learningRate * gradW_0 / Math.sqrt(W_0_hist[c] + EPS);
				}
			}

			if (it % 10 == 0) {
				LOGGER.debug("Iteration {}/{}", it, this.maxIter);

				long currTime = System.currentTimeMillis();
				if (currTime - beginTime > this.timeout.milliseconds()) {
					LOGGER.debug("Stopping training due to timeout.");
					break;
				}
			}
		}
		LOGGER.debug("Finished training.");

		// Update model
		this.model.setS(S);
		this.model.setW(W);
		this.model.setW_0(W_0);
		this.model.setC(this.C);
		this.model.setMinShapeLength(this.minShapeLength);

		return this.model;
	}

	/**
	 * Function to calculate the soft-minimum function which is a differentiable
	 * approximation of the minimum distance matrix given in the paper in section
	 * 3.1.4.
	 * 
	 * @param S
	 *            The tensor storing the shapelets for different scales
	 * @param minShapeLength
	 *            The minimum shape length
	 * @param r
	 *            The number of scale to look at
	 * @param instance
	 *            The instance time series vector
	 * @param k
	 *            The index of the shapelet to look at
	 * @param Q
	 *            The number of attributes (time series length)
	 * @param alpha
	 *            Parameter to control the desired precision of the M_hat
	 *            approximation
	 * @return Returns the approximation of the minimum distance of the instance and
	 *         the shapelet given by the parameters <code>r</code> and
	 *         <code>k</code>.
	 */
	public static double calculateM_hat(final double[][][] S, final int minShapeLength, final int r,
			final double[] instance, final int k, final int Q, final double alpha) {
		double nominator = 0;
		double denominator = 0;
		for (int j = 0; j < getNumberOfSegments(Q, minShapeLength, r); j++) {
			double D = calculateD(S, minShapeLength, r, instance, k, j);
			double expD = Math.exp(alpha * D);
			nominator += D * expD;
			denominator += expD;
		}
		denominator = denominator == 0d ? EPS : denominator;
		return nominator / denominator;
	}

	/**
	 * Function to calculate the distance between the <code>j</code>-th segment of
	 * the given time series <code>instance</code> and the <code>k</code>-th
	 * shapelet stored in the shapelet tensor <code>S</code>.
	 * 
	 * @param S
	 *            The tensor storing the shapelets for different scales
	 * @param minShapeLength
	 *            The minimum shape length
	 * @param r
	 *            The number of scale to look at
	 * @param instance
	 *            The instance time series vector
	 * @param k
	 *            The index of the shapelet to look at
	 * @param j
	 *            The segment of the instance time series to look at
	 * @return Returns the minimum distance of the <code>j</code>-th segment of the
	 *         instance and the shapelet given by the parameters <code>r</code>,
	 *         <code>k</code> and <code>j</code>.
	 */
	public static double calculateD(final double[][][] S, final int minShapeLength, final int r,
			final double[] instance, final int k, final int j) {

		double result = 0;
		for (int l = 0; l < (r + 1) * minShapeLength; l++) {
			result += Math.pow(instance[j + l] - S[r][k][l], 2);
		}
		return result / (double) ((r + 1) * minShapeLength);
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
	public TimeSeriesDataset getInput() {
		return this.input;
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
	public void setNumCPUs(int numberOfCPUs) {
		LOGGER.warn(
				"Multithreading is not supported for LearnShapelets yet. Therefore, the number of CPUs is not considered.");
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public int getNumCPUs() {
		LOGGER.warn(
				"Multithreading is not supported for LearnShapelets yet. Therefore, the number of CPUs is not considered.");
		return 1;
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
	public AlgorithmEvent nextWithException() {
		throw new UnsupportedOperationException("The operation to be performed is not supported.");
	}

	/**
	 * Returns the number of segments which are available for a instance with
	 * <code>Q</code> attributes for a given scale <code>r</code> and a minimum
	 * shape length <code>minShapeLength</code>.
	 * 
	 * @param Q
	 *            Number of attributes of an instance
	 * @param minShapeLength
	 *            Minimum shapelet length
	 * @param r
	 *            Scale to be looked at
	 * @return Returns the number of segments which can be looked at for an instance
	 *         with <code>Q</code> time series attributes
	 */
	public static int getNumberOfSegments(final int Q, final int minShapeLength, final int r) {
		return Q - (r + 1) * minShapeLength;
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public IAlgorithmConfig getConfig() {
		throw new UnsupportedOperationException("The operation to be performed is not supported.");
	}

	/**
	 * Getter for {@link LearnShapeletsAlgorithm#estimateK}.
	 * 
	 * @return the estimateK
	 */
	public boolean isEstimateK() {
		return estimateK;
	}

	/**
	 * Setter for {@link LearnShapeletsAlgorithm#estimateK}.
	 * 
	 * @param estimateK
	 *            the estimateK to set
	 */
	public void setEstimateK(boolean estimateK) {
		this.estimateK = estimateK;
	}

}
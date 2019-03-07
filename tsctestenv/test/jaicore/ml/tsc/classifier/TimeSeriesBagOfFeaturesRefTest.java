package jaicore.ml.tsc.classifier;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Map;
import java.util.Random;

import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import jaicore.ml.core.exception.EvaluationException;
import jaicore.ml.core.exception.PredictionException;
import jaicore.ml.core.exception.TrainingException;
import jaicore.ml.tsc.classifier.trees.TimeSeriesBagOfFeaturesClassifier;
import jaicore.ml.tsc.exceptions.TimeSeriesLoadingException;
import timeseriesweka.classifiers.TSBF;

/**
 * Reference tests for {@link TimeSeriesBagOfFeaturesClassifier} objects.
 * 
 * @author Julian Lienen
 *
 */
@SuppressWarnings("unused")
public class TimeSeriesBagOfFeaturesRefTest {
	private static final Logger LOGGER = LoggerFactory.getLogger(TimeSeriesBagOfFeaturesRefTest.class);

	private static final String UNIVARIATE_PREFIX = "D:\\Data\\TSC\\UnivariateTSCProblems\\";

	private static final String CAR_TRAIN = UNIVARIATE_PREFIX + "Car\\Car_TRAIN.arff";
	private static final String CAR_TEST = UNIVARIATE_PREFIX + "Car\\Car_TEST.arff";

	private static final String BEEF_TRAIN = UNIVARIATE_PREFIX + "Beef\\Beef_TRAIN.arff";
	private static final String BEEF_TEST = UNIVARIATE_PREFIX + "Beef\\Beef_TEST.arff";

	private static final String ARROW_HEAD_TRAIN = UNIVARIATE_PREFIX + "ArrowHead\\ArrowHead\\ArrowHead_TRAIN.arff";
	private static final String ARROW_HEAD_TEST = UNIVARIATE_PREFIX + "ArrowHead\\ArrowHead\\ArrowHead_TEST.arff";

	private static final String ITALY_POWER_DEMAND_TRAIN = UNIVARIATE_PREFIX
			+ "ItalyPowerDemand\\ItalyPowerDemand_TRAIN.arff";
	private static final String ITALY_POWER_DEMAND_TEST = UNIVARIATE_PREFIX
			+ "ItalyPowerDemand\\ItalyPowerDemand_TEST.arff";

	private static final String RACKET_SPORTS_TRAIN = UNIVARIATE_PREFIX + "RacketSports\\RacketSports_TRAIN.arff";
	private static final String RACKET_SPORTS_TEST = UNIVARIATE_PREFIX + "RacketSports\\RacketSports_TEST.arff";

	private static final String SYNTHETIC_CONTROL_TRAIN = UNIVARIATE_PREFIX
			+ "\\SyntheticControl\\SyntheticControl_TRAIN.arff";
	private static final String SYNTHETIC_CONTROL_TEST = UNIVARIATE_PREFIX
			+ "\\SyntheticControl\\SyntheticControl_TEST.arff";

	private static final String COMPUTERS_TRAIN = UNIVARIATE_PREFIX + "\\Computers\\Computers_TRAIN.arff";
	private static final String COMPUTERS_TEST = UNIVARIATE_PREFIX + "\\Computers\\Computers_TEST.arff";

	@Test
	public void testClassifier() throws FileNotFoundException, EvaluationException, TrainingException,
			PredictionException, IOException, TimeSeriesLoadingException, ClassNotFoundException {

		// TODO: Change this?
		org.apache.log4j.Logger.getLogger("jaicore").setLevel(org.apache.log4j.Level.INFO);

		// int seed = 42;
		// int numBins = 20; // As in the reference implementation
		// int numFolds = 20; // As in the reference implementation
		// double zProp = 1; // As in the reference implementation
		// int minIntervalLength = 5; // As in the reference implementation

		double currBest = 0;
		double[] z = new double[] { 0.1, 0.25, 0.5, 0.75 };

		int numTotalIterations = 1000;
		Random seedRandom = new Random(42);
		for (int i = 0; i < numTotalIterations; i++) {
			int seed = seedRandom.nextInt(100);

			Random random = new Random(seed);
			int numBins = 10; // 1 + random.nextInt(20); // As in the reference implementation
			int numFolds = 10; // 3 + random.nextInt(15); // As in the reference implementation
			double zProp = z[i % z.length];// 0.01 + random.nextDouble(); // As in the reference implementation
			if (zProp > 1)
				zProp = 1d;
			int minIntervalLength = 5; // 2 + random.nextInt(10); // As in the reference implementation

			TimeSeriesBagOfFeaturesClassifier ownClf = new TimeSeriesBagOfFeaturesClassifier(seed, numBins, numFolds,
					zProp, minIntervalLength);

			TSBF refClf = new TSBF();
			refClf.seedRandom(seed);

			Map<String, Object> result = SimplifiedTSClassifierTest.compareClassifiers(refClf, ownClf, seed, null, null,
					new File(ITALY_POWER_DEMAND_TRAIN), new File(ITALY_POWER_DEMAND_TEST));
			if (((double) result.get("accuracy")) > currBest) {
				currBest = ((double) result.get("accuracy"));
				LOGGER.info(
						"New best score {} with numBins {}, numFolds {}, zProp {} and minIntervalLength {} (seed {}).",
						currBest, numBins, numFolds, zProp, minIntervalLength, seed);
			}

			if (i % 100 == 0)
				LOGGER.info("{}/{}", i, numTotalIterations);
		}

		LOGGER.info("Final best score: {}", currBest);

		// System.out.println("Ref clf parameters: " + refClf.getParameters());
		// System.out.println(result.toString());
	}
}

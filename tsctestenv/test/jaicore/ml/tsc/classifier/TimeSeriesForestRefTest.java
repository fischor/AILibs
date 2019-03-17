package jaicore.ml.tsc.classifier;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import org.apache.log4j.Level;
import org.junit.Test;

import jaicore.basic.TimeOut;
import jaicore.ml.core.exception.EvaluationException;
import jaicore.ml.core.exception.PredictionException;
import jaicore.ml.core.exception.TrainingException;
import jaicore.ml.tsc.classifier.trees.TimeSeriesForestClassifier;
import jaicore.ml.tsc.exceptions.TimeSeriesLoadingException;
import timeseriesweka.classifiers.TSF;

/**
 * Reference tests for the {@link TimeSeriesForestClassifier}.
 * 
 * @author Julian Lienen
 *
 */
@SuppressWarnings("unused")
public class TimeSeriesForestRefTest {
	private static final double EPS_DELTA = 0.000001;

	private static final String UNIVARIATE_PREFIX = "C:\\Users\\Julian\\Downloads\\UnivariateTSCProblems\\";

	private static final String CAR_TRAIN = "C:\\Users\\Julian\\Downloads\\UnivariateTSCProblems\\Car\\Car_TRAIN.arff";
	private static final String CAR_TEST = "C:\\Users\\Julian\\Downloads\\UnivariateTSCProblems\\Car\\Car_TEST.arff";

	private static final String ARROW_HEAD_TRAIN = "C:\\Users\\Julian\\Downloads\\UnivariateTSCProblems\\ArrowHead\\ArrowHead\\ArrowHead_TRAIN.arff";
	private static final String ARROW_HEAD_TEST = "C:\\Users\\Julian\\Downloads\\UnivariateTSCProblems\\ArrowHead\\ArrowHead\\ArrowHead_TEST.arff";

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

	private static final String BEEF_TRAIN = UNIVARIATE_PREFIX + "Beef\\Beef_TRAIN.arff";
	private static final String BEEF_TEST = UNIVARIATE_PREFIX + "Beef\\Beef_TEST.arff";

	@Test
	public void testClassifier() throws FileNotFoundException, EvaluationException, TrainingException,
			PredictionException, IOException, TimeSeriesLoadingException {

		org.apache.log4j.Logger.getLogger("jaicore").setLevel(Level.DEBUG);

		int seed = 42;
		int numTrees = 500;
		// Ref classifier uses no depth limit
		int maxDepth = 1000;
		int numCPUs = 1;

		TimeSeriesForestClassifier ownClf = new TimeSeriesForestClassifier(numTrees, maxDepth, seed, false, numCPUs,
				new TimeOut(Integer.MAX_VALUE, TimeUnit.SECONDS));

		TSF refClf = new TSF(seed);
		refClf.setNumTrees(numTrees);

		Map<String, Object> result = SimplifiedTSClassifierTest.compareClassifiers(refClf, ownClf, seed, null, null,
				new File(ITALY_POWER_DEMAND_TRAIN), new File(ITALY_POWER_DEMAND_TEST));

		System.out.println("Ref clf parameters: " + refClf.getParameters());
		System.out.println(result.toString());
	}
}

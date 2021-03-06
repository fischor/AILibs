package jaicore.search.testproblems.knapsack;

import jaicore.basic.algorithm.AlgorithmProblemTransformer;
import jaicore.search.probleminputs.GraphSearchWithPathEvaluationsInput;
import jaicore.search.testproblems.knapsack.KnapsackProblem.KnapsackNode;

public class KnapsackToGraphSearchProblemInputReducer implements AlgorithmProblemTransformer<KnapsackProblem, GraphSearchWithPathEvaluationsInput<KnapsackNode, String, Double>> {

	@Override
	public GraphSearchWithPathEvaluationsInput<KnapsackNode, String, Double> transform(KnapsackProblem problem) {
		return new GraphSearchWithPathEvaluationsInput<>(problem.getGraphGenerator(), problem.getSolutionEvaluator());

	}
}

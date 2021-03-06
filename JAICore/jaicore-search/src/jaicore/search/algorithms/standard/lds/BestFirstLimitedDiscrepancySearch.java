package jaicore.search.algorithms.standard.lds;

import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeoutException;
import java.util.stream.Collectors;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.eventbus.Subscribe;

import jaicore.basic.ILoggingCustomizable;
import jaicore.basic.algorithm.AlgorithmExecutionCanceledException;
import jaicore.basic.algorithm.events.AlgorithmEvent;
import jaicore.basic.algorithm.exceptions.AlgorithmException;
import jaicore.search.algorithms.standard.bestfirst.StandardBestFirst;
import jaicore.search.algorithms.standard.bestfirst.events.SuccessorComputationCompletedEvent;
import jaicore.search.algorithms.standard.bestfirst.nodeevaluation.INodeEvaluator;
import jaicore.search.core.interfaces.AOptimalPathInORGraphSearch;
import jaicore.search.model.other.EvaluatedSearchGraphPath;
import jaicore.search.model.travesaltree.Node;
import jaicore.search.probleminputs.GraphSearchWithNodeRecommenderInput;
import jaicore.search.probleminputs.GraphSearchWithSubpathEvaluationsInput;

/**
 * This class conducts a limited discrepancy search by running a best first algorithm with list-based node evaluations Since the f-values are lists too, we do not simply extend BestFirst but rather forward all commands to it
 *
 * @author fmohr
 *
 * @param <T>
 * @param <A>
 * @param <V>
 */
public class BestFirstLimitedDiscrepancySearch<T, A, V extends Comparable<V>> extends AOptimalPathInORGraphSearch<GraphSearchWithNodeRecommenderInput<T, A>, T, A, V, Node<T, NodeOrderList>, A> {

	private Logger logger = LoggerFactory.getLogger(BestFirstLimitedDiscrepancySearch.class);
	private String loggerName;

	private final StandardBestFirst<T, A, NodeOrderList> bestFirst;

	private class OrderListNumberComputer implements INodeEvaluator<T, NodeOrderList> {
		private final Comparator<T> heuristic;
		private final Map<Node<T, ?>, List<T>> childOrdering = new HashMap<>();

		public OrderListNumberComputer(final Comparator<T> heuristic) {
			super();
			this.heuristic = heuristic;
		}

		@Override
		public NodeOrderList f(final Node<T, ?> node) {
			NodeOrderList list = new NodeOrderList();
			Node<T, ?> parent = node.getParent();
			if (parent == null) {
				return list;
			}

			/* add the label sequence of the parent to this node*/
			list.addAll((NodeOrderList) parent.getInternalLabel());
			list.add(this.childOrdering.get(parent).indexOf(node.getPoint()));
			return list;
		}

		@Subscribe
		public void receiveSuccessorsCreatedEvent(final SuccessorComputationCompletedEvent<T, ?> successorDescriptions) {
			List<T> successors = successorDescriptions.getSuccessorDescriptions().stream().map(n -> n.getTo()).sorted(this.heuristic).collect(Collectors.toList());
			this.childOrdering.put(successorDescriptions.getNode(), successors);
		}
	}

	public BestFirstLimitedDiscrepancySearch(final GraphSearchWithNodeRecommenderInput<T, A> problem) {
		super(problem);
		OrderListNumberComputer nodeEvaluator = new OrderListNumberComputer(problem.getRecommender());
		this.bestFirst = new StandardBestFirst<>(new GraphSearchWithSubpathEvaluationsInput<>(problem.getGraphGenerator(), nodeEvaluator));
		this.bestFirst.registerListener(nodeEvaluator);
	}

	@Override
	public void cancel() {
		super.cancel();
		this.bestFirst.cancel();
	}

	@Override
	public void registerListener(final Object listener) {
		this.bestFirst.registerListener(listener);
	}

	@Override
	public void setNumCPUs(final int numberOfCPUs) {
		super.setNumCPUs(numberOfCPUs);
		this.bestFirst.setNumCPUs(numberOfCPUs);
	}

	@Override
	public AlgorithmEvent nextWithException() throws InterruptedException, AlgorithmExecutionCanceledException, TimeoutException, AlgorithmException  {
		return this.bestFirst.nextWithException();
	}

	@Override
	public EvaluatedSearchGraphPath<T, A, V> call() throws InterruptedException, AlgorithmExecutionCanceledException, TimeoutException, AlgorithmException  {
		EvaluatedSearchGraphPath<T, A, NodeOrderList> solution = this.bestFirst.call();
		EvaluatedSearchGraphPath<T, A, V> modifiedSolution = new EvaluatedSearchGraphPath<T, A, V>(solution.getNodes(), solution.getEdges(), null);
		return modifiedSolution;
	}

	@Override
	public String getLoggerName() {
		return this.loggerName;
	}

	@Override
	public void setLoggerName(final String name) {
		this.logger.info("Switching logger from {} to {}", this.logger.getName(), name);
		this.loggerName = name;
		this.logger = LoggerFactory.getLogger(name);
		this.logger.info("Activated logger {} with name {}", name, this.logger.getName());
		if (this.bestFirst instanceof ILoggingCustomizable) {
			this.bestFirst.setLoggerName(name + ".bestfirst");
		}
		super.setLoggerName(this.loggerName + "._orgraphsearch");
	}
}

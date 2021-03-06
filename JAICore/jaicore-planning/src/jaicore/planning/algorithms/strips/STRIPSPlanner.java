package jaicore.planning.algorithms.strips;

import java.util.Collection;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.concurrent.TimeoutException;
import java.util.stream.Collectors;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import jaicore.basic.ILoggingCustomizable;
import jaicore.basic.algorithm.AlgorithmExecutionCanceledException;
import jaicore.basic.algorithm.events.AlgorithmEvent;
import jaicore.basic.algorithm.exceptions.AlgorithmException;
import jaicore.basic.sets.SetUtil;
import jaicore.graphvisualizer.TooltipGenerator;
import jaicore.graphvisualizer.gui.VisualizationWindow;
import jaicore.logic.fol.structure.VariableParam;
import jaicore.planning.EvaluatedPlan;
import jaicore.planning.algorithms.IPlanningAlgorithm;
import jaicore.planning.algorithms.events.PlanFoundEvent;
import jaicore.planning.graphgenerators.strips.forward.StripsForwardPlanningNode;
import jaicore.planning.graphgenerators.strips.forward.StripsTooltipGenerator;
import jaicore.planning.model.core.Operation;
import jaicore.planning.model.core.Plan;
import jaicore.planning.model.strips.StripsAction;
import jaicore.planning.model.strips.StripsOperation;
import jaicore.planning.model.strips.StripsPlanningProblem;
import jaicore.search.algorithms.standard.bestfirst.BestFirst;
import jaicore.search.algorithms.standard.bestfirst.nodeevaluation.INodeEvaluator;
import jaicore.search.core.interfaces.GraphGenerator;
import jaicore.search.core.interfaces.IPathInORGraphSearch;
import jaicore.search.model.other.EvaluatedSearchGraphPath;
import jaicore.search.model.travesaltree.Node;
import jaicore.search.probleminputs.GraphSearchWithSubpathEvaluationsInput;

public class STRIPSPlanner<V extends Comparable<V>>
		extends IPlanningAlgorithm<StripsPlanningProblem, EvaluatedPlan<StripsAction, V>, V> {

	/* logging */
	private Logger logger = LoggerFactory.getLogger(BestFirst.class);
	private String loggerName;

	private final IPathInORGraphSearch<GraphSearchWithSubpathEvaluationsInput<StripsForwardPlanningNode, String, V>, EvaluatedSearchGraphPath<StripsForwardPlanningNode, String, V>, StripsForwardPlanningNode, String, Node<StripsForwardPlanningNode, V>, String> search;
	private final INodeEvaluator<StripsForwardPlanningNode, V> nodeEvaluator;
	private final STRIPSForwardSearchReducer reducer = new STRIPSForwardSearchReducer();

	/* state of the algorithm */
	private boolean visualize = false;
	private final GraphGenerator<StripsForwardPlanningNode, String> graphGenerator;

	public STRIPSPlanner(final StripsPlanningProblem problem,
			final INodeEvaluator<StripsForwardPlanningNode, V> nodeEvaluator) {
		super(problem);
		this.nodeEvaluator = nodeEvaluator;

		/* conduct some consistency checks */
		if (!problem.getInitState().getVariableParams().isEmpty()) {
			throw new IllegalArgumentException(
					"The initial state contains variable parameters but must only contain constants!\nList of found variables: "
							+ problem.getInitState().getVariableParams().stream().map(n -> "\n\t" + n.getName())
									.collect(Collectors.joining()));
		}
		if (!problem.getGoalState().getVariableParams().isEmpty()) {
			throw new IllegalArgumentException(
					"The goal state contains variable parameters but must only contain constants!\nList of found variables: "
							+ problem.getGoalState().getVariableParams().stream().map(n -> "\n\t" + n.getName())
									.collect(Collectors.joining()));
		}

		/*
		 * check that every operation has only arguments in its preconditions, add lists
		 * and delete lists, that are also explicitly defined in the param list
		 */
		for (Operation o : problem.getDomain().getOperations()) {
			StripsOperation so = (StripsOperation) o;
			Collection<VariableParam> undeclaredParamsInPrecondition = SetUtil
					.difference(so.getPrecondition().getVariableParams(), so.getParams());
			if (!undeclaredParamsInPrecondition.isEmpty()) {
				throw new IllegalArgumentException("The precondition of operation " + so.getName()
						+ " contains variables that are not defined in the parameter list: "
						+ undeclaredParamsInPrecondition);
			}
			Collection<VariableParam> undeclaredParamsInAddList = SetUtil
					.difference(so.getAddList().getVariableParams(), so.getParams());
			if (!undeclaredParamsInAddList.isEmpty()) {
				throw new IllegalArgumentException("The add list of operation " + so.getName()
						+ " contains variables that are not defined in the parameter list: "
						+ undeclaredParamsInAddList);
			}
			Collection<VariableParam> undeclaredParamsInDelList = SetUtil
					.difference(so.getDeleteList().getVariableParams(), so.getParams());
			if (!undeclaredParamsInDelList.isEmpty()) {
				throw new IllegalArgumentException("The del list of operation " + so.getName()
						+ " contains variables that are not defined in the parameter list: "
						+ undeclaredParamsInDelList);
			}
		}

		/* logging problem */
		this.logger.info(
				"Initializing planner for the following problem:\n\tOperations:{}\n\tInitial State: {}\n\tGoal State: {}",
				problem.getDomain().getOperations().stream()
						.map(o -> "\n\t - " + o.getName() + "\n\t\tParams: " + o.getParams() + "\n\t\tPre: "
								+ o.getPrecondition() + "\n\t\tAdd: " + ((StripsOperation) o).getAddList()
								+ "\n\t\tDel: " + ((StripsOperation) o).getDeleteList())
						.collect(Collectors.joining()),
				problem.getInitState(), problem.getGoalState());

		/* create search algorithm */
		this.graphGenerator = this.reducer.transform(problem);
		GraphSearchWithSubpathEvaluationsInput<StripsForwardPlanningNode, String, V> searchProblem = new GraphSearchWithSubpathEvaluationsInput<>(
				this.graphGenerator, nodeEvaluator);
		this.search = new BestFirst<GraphSearchWithSubpathEvaluationsInput<StripsForwardPlanningNode, String, V>, StripsForwardPlanningNode, String, V>(
				searchProblem);
	}

	@Override
	public AlgorithmEvent nextWithException()
			throws AlgorithmExecutionCanceledException, InterruptedException, TimeoutException, AlgorithmException {
		switch (this.getState()) {
		case created:
			this.setLoggerOfSearch();
			this.search.setTimeout(this.getTimeout());
			if (this.visualize) {
				VisualizationWindow<Node<StripsForwardPlanningNode, V>, String> w = new VisualizationWindow<>(
						this.search);
				TooltipGenerator<StripsForwardPlanningNode> tt = new StripsTooltipGenerator<>();
				w.setTooltipGenerator(n -> tt.getTooltip(((Node<StripsForwardPlanningNode, V>) n).getPoint()));
			}
			return activate();
		case active:

			/* invoke next step in search algorithm */
			if (!this.search.hasNext()) {
				return terminate();
			}
			try {
				EvaluatedSearchGraphPath<StripsForwardPlanningNode, String, V> nextSolution = this.search
						.nextSolutionCandidate();
				Plan<StripsAction> plan = this.reducer.getPlan(nextSolution.getNodes());
				EvaluatedPlan<StripsAction, V> evaluatedPlan = new EvaluatedPlan<>(plan, nextSolution.getScore());
				this.updateBestSeenSolution(evaluatedPlan);
				return new PlanFoundEvent<>(evaluatedPlan);
			} catch (NoSuchElementException e) {
				return terminate();
			}

		default:
			throw new IllegalStateException("Cannot handle algorithm state " + this.getState());
		}
	}

	public void enableVisualization() {
		this.visualize = true;
	}

	@Override
	public void cancel() {
		super.cancel();
		if (this.search != null) {
			this.search.cancel();
		}
	}

	@Override
	public Map<String, Object> getAnnotationsOfSolution(final EvaluatedPlan<StripsAction, V> solution) {
		// TODO Auto-generated method stub
		return null;
	}

	private void setLoggerOfSearch() {
		if (this.search != null && loggerName != null) {
			if (this.search instanceof ILoggingCustomizable) {
				this.logger.info("Switching logger of search to {}.search", this.getLoggerName());
				((ILoggingCustomizable) this.search).setLoggerName(this.getLoggerName() + ".search");
			} else {
				this.logger.info("The search is of class {}, which is not logging customizable.",
						this.search.getClass());
			}
		} else
			this.logger.info("Not yet setting logger of search, since search has not yet been configured.");
	}

	public GraphGenerator<StripsForwardPlanningNode, String> getGraphGenerator() {
		return this.graphGenerator;
	}

	@Override
	public void setLoggerName(final String name) {
		if (name == null)
			throw new IllegalArgumentException("Logger name must not be set to null.");
		this.logger.info("Switching logger from {} to {}", this.logger.getName(), name);
		this.loggerName = name;
		this.logger = LoggerFactory.getLogger(name);
		this.logger.info("Activated logger {} with name {}", name, this.logger.getName());
		if (this.nodeEvaluator instanceof ILoggingCustomizable) {
			((ILoggingCustomizable) this.nodeEvaluator).setLoggerName(name + ".nodeeval");
		}
		this.setLoggerOfSearch();
		super.setLoggerName(this.loggerName + "._planningalgorithm");
	}

	public String getLoggerName() {
		return loggerName;
	}
}

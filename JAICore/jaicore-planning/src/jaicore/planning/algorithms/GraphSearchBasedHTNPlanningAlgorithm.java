package jaicore.planning.algorithms;

import java.util.Iterator;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.eventbus.EventBus;

import jaicore.basic.ILoggingCustomizable;
import jaicore.basic.algorithm.AlgorithmEvent;
import jaicore.basic.algorithm.AlgorithmFinishedEvent;
import jaicore.basic.algorithm.AlgorithmInitializedEvent;
import jaicore.basic.algorithm.AlgorithmState;
import jaicore.basic.algorithm.IAlgorithm;
import jaicore.basic.algorithm.IAlgorithmListener;
import jaicore.graph.IGraphAlgorithmListener;
import jaicore.planning.EvaluatedSearchGraphBasedPlan;
import jaicore.planning.algorithms.events.PlanFoundEvent;
import jaicore.planning.algorithms.forwarddecomposition.ForwardDecompositionHTNPlanner;
import jaicore.planning.graphgenerators.IPlanningGraphGeneratorDeriver;
import jaicore.planning.graphgenerators.task.tfd.TFDNode;
import jaicore.planning.model.core.Action;
import jaicore.planning.model.core.Operation;
import jaicore.planning.model.core.Plan;
import jaicore.planning.model.task.IHTNPlanningProblem;
import jaicore.planning.model.task.stn.Method;
import jaicore.search.core.interfaces.IGraphSearch;
import jaicore.search.core.interfaces.IGraphSearchFactory;
import jaicore.search.model.other.EvaluatedSearchGraphPath;
import jaicore.search.model.probleminputs.builders.SearchProblemInputBuilder;

public class GraphSearchBasedHTNPlanningAlgorithm<PA extends Action, P extends IHTNPlanningProblem<?,?,PA>, ISearch, OSearch, NSrc, ASrc, V extends Comparable<V>, NSearch, ASearch, L extends IAlgorithmListener> implements IAlgorithm<P, EvaluatedSearchGraphBasedPlan<PA, V, NSrc>, L>, ILoggingCustomizable {
	
	/* logging and communication */
	private String loggerName;
	private Logger logger = LoggerFactory.getLogger(ForwardDecompositionHTNPlanner.class);
	private final EventBus eventBus = new EventBus();
	
	/* algorithm inputs */
	private final P planningProblem;
	private final IPlanningGraphGeneratorDeriver<?,?,PA,P, NSrc, ASrc> problemTransformer;
	private IGraphSearch<ISearch, OSearch, NSrc, ASrc, V, NSearch, ASearch, IGraphAlgorithmListener<NSearch, ASearch>> search;

	/* state of the algorithm */
	private AlgorithmState state = AlgorithmState.created;
	private boolean canceled = false;
	
	public GraphSearchBasedHTNPlanningAlgorithm(P problem,
			IPlanningGraphGeneratorDeriver<?,?,PA,P, NSrc, ASrc> problemTransformer,
			IGraphSearchFactory<ISearch, OSearch, NSrc, ASrc, V, NSearch, ASearch, IGraphAlgorithmListener<NSearch, ASearch>> searchFactory,
			SearchProblemInputBuilder<NSrc, ASrc, ISearch> searchProblemBuilder) {
		
		this.planningProblem = problem;
		this.problemTransformer = problemTransformer;
		
		/* set the problem in the search factory */
		searchProblemBuilder.setGraphGenerator(problemTransformer.transform(problem));
		searchFactory.setProblemInput(searchProblemBuilder.build());
		search = searchFactory.getAlgorithm();
	}
	
	public List<Action> getPlan(List<TFDNode> path) {
		return path.stream().filter(n -> n.getAppliedAction() != null).map(n -> n.getAppliedAction()).collect(Collectors.toList());
	}

	@Override
	public void cancel() {
		this.canceled = true;
		getSearch().cancel();
	}

	@Override
	public void setLoggerName(String name) {
		this.logger.info("Switching logger from {} to {}", logger.getName(), name);
		this.loggerName = name;
		logger = LoggerFactory.getLogger(name);
		this.logger.info("Activated logger {} with name {}", name, logger.getName());
	}

	@Override
	public String getLoggerName() {
		return loggerName;
	}

	@Override
	public Iterator<AlgorithmEvent> iterator() {
		return this;
	}

	@Override
	public boolean hasNext() {
		return state != AlgorithmState.inactive;
	}

	@Override
	public AlgorithmEvent next() {

		logger.debug("I'm being asked whether there is a next solution.");
		try {
			switch (state) {
			case created: {
				logger.info("Starting HTN planning process.");
				if (logger.isDebugEnabled()) {
					StringBuilder opSB = new StringBuilder();
					for (Operation op : planningProblem.getDomain().getOperations()) {
						opSB.append("\n\t\t");
						opSB.append(op);
					}
					StringBuilder methodSB = new StringBuilder();
					for (Method method : planningProblem.getDomain().getMethods()) {
						methodSB.append("\n\t\t");
						methodSB.append(method);
					}
					logger.debug("The HTN problem is defined as follows:\n\tOperations:{}\n\tMethods:{}", opSB.toString(), methodSB.toString());
				}

				if (loggerName != null && loggerName.length() > 0 && search instanceof ILoggingCustomizable) {
					logger.info("Customizing logger of search with {}", loggerName);
					((ILoggingCustomizable) search).setLoggerName(loggerName + ".search");
				}
				state = AlgorithmState.active;
				return new AlgorithmInitializedEvent();
			}
			case active: {
				if (canceled)
					throw new IllegalStateException("The planner has already been canceled. Cannot compute more plans.");
				logger.info("Starting/continuing search for next plan.");
				EvaluatedSearchGraphPath<NSrc, ASrc, V> solution = search.nextSolution();
				if (solution == null) {
					logger.info("No more solutions will be found. Terminating algorithm.");
					state = AlgorithmState.inactive;
					return new AlgorithmFinishedEvent();
				}
				logger.info("Next solution found.");
				List<NSrc> solutionPath = solution.getNodes();
				Plan<PA> plan = problemTransformer.getPlan(solutionPath);
				PlanFoundEvent<PA, V> event = new PlanFoundEvent<>(new EvaluatedSearchGraphBasedPlan<>(plan.getActions(), solution.getScore(), solution));
				eventBus.post(event);
				return event;
			}
			default:
				throw new IllegalStateException("Don't know what to do in state " + state);
			}
		} catch (Exception e) {
			throw new RuntimeException(e);
		}
	}

	@Override
	public P getInput() {
		return planningProblem;
	}

	@Override
	public void registerListener(L listener) {
		eventBus.register(listener);
	}

	@Override
	public int getNumCPUs() {
		return search.getNumCPUs();
	}

	@Override
	public void setNumCPUs(int numberOfCPUs) {
		search.setNumCPUs(numberOfCPUs);
	}

	@Override
	public EvaluatedSearchGraphBasedPlan<PA, V, NSrc> call() throws Exception {
		return null;
	}

	@Override
	public void setTimeout(int timeout, TimeUnit timeUnit) {
		// TODO Auto-generated method stub

	}

	@Override
	public int getTimeout() {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public TimeUnit getTimeoutUnit() {
		// TODO Auto-generated method stub
		return null;
	}

	public IGraphSearch<ISearch, OSearch, NSrc, ASrc, V, NSearch, ASearch, IGraphAlgorithmListener<NSearch, ASearch>> getSearch() {
		return search;
	}
}
package hex.genmodel.algos.tree;

import hex.genmodel.PredictContributions;
import hex.genmodel.attributes.parameters.FeatureContribution;
import hex.genmodel.utils.ArrayUtils;

public abstract class ContributionsPredictor<E> implements PredictContributions {
  private final int _ncontribs;
  private final String[] _contribution_names;
  private final TreeSHAPPredictor<E> _treeSHAPPredictor;
  private final ContributionComposer contributionComposer = new ContributionComposer();

  private static final ThreadLocal<Object> _workspace = new ThreadLocal<>();

  public ContributionsPredictor(int ncontribs, String[] featureContributionNames, TreeSHAPPredictor<E> treeSHAPPredictor) {
    _ncontribs = ncontribs;
    _contribution_names = ArrayUtils.append(featureContributionNames, "BiasTerm");
    _treeSHAPPredictor = treeSHAPPredictor;
  }

  @Override
  public final String[] getContributionNames() {
    return _contribution_names;
  }

  public final float[] calculateContributions(double[] input) {
    float[] contribs = new float[_ncontribs];
    _treeSHAPPredictor.calculateContributions(toInputRow(input), contribs, 0, -1, getWorkspace());
    return getContribs(contribs);
  }

  protected abstract E toInputRow(double[] input);

  public float[] getContribs(float[] contribs) {
    return contribs;
  }

  private Object getWorkspace() {
    Object workspace = _workspace.get();
    if (workspace == null) {
      workspace = _treeSHAPPredictor.makeWorkspace();
      _workspace.set(workspace);
    }
    return workspace;
  }

  @Override
  public FeatureContribution[] calculateContributions(double[] input, int topN, int topBottomN, boolean abs) {
    float[] contributions = calculateContributions(input);
    Integer[] sorted = contributionComposer.composeContributions(contributions, ArrayUtils.interval(0, _contribution_names.length - 1, 1), topN, topBottomN, abs);
    FeatureContribution[] out = new FeatureContribution[sorted.length];
    for (int i = 0; i < sorted.length; i++) {
      out[i] = new FeatureContribution(_contribution_names[sorted[i]], contributions[sorted[i]]);
    }
    return out;
  }
}


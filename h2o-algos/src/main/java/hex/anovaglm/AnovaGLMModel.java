package hex.anovaglm;

import hex.DataInfo;
import hex.Model;
import hex.ModelCategory;
import hex.ModelMetrics;
import hex.deeplearning.DeepLearningModel;
import hex.glm.GLM;
import water.AutoBuffer;
import water.Futures;
import water.Key;
import water.Keyed;
import water.fvec.Frame;

import java.io.Serializable;

import static hex.glm.GLMModel.GLMParameters.*;


public class AnovaGLMModel extends Model<AnovaGLMModel, AnovaGLMModel.AnovaGLMParameters, AnovaGLMModel.AnovaGLMOutput>{

  /**
   * Full constructor
   *
   * @param selfKey
   * @param parms
   * @param output
   */
  public AnovaGLMModel(Key<AnovaGLMModel> selfKey, AnovaGLMParameters parms, AnovaGLMOutput output) {
    super(selfKey, parms, output);
  }

  @Override
  public ModelMetrics.MetricBuilder makeMetricBuilder(String[] domain) {
    return null;
  }

  @Override
  protected double[] score0(double[] data, double[] preds) {
    return new double[0];
  }

  public static class AnovaGLMParameters extends Model.Parameters {
    public boolean _standardize = false;
    Family _family = Family.gaussian; // default family, also takes tweedie
    public Link _link;
    public Solver _solver = Solver.IRLSM;
    public double[] _lambda = new double[]{0};  // no regularization needed
    public boolean _lambda_search = false;      // no lambda search
    public double _tweedie_variance_power;
    public double _tweedie_link_power;
    public double _theta;
    public double _invTheta;
    public Serializable _missing_values_handling = MissingValuesHandling.MeanImputation;
    public boolean _compute_p_values = true;
    public boolean _remove_collinear_columns = true;
    public int _nfolds = 0; // disable cross-validation
    public Key<Frame> _plug_values = null;
    public boolean _save_transformed_framekeys = false; // for debugging, save the transformed predictors/interaction

    @Override
    public String algoName() {
      return "AnovaGLM";
    }

    @Override
    public String fullName() {
      return "Anova for Generalized Linear Model";
    }

    @Override
    public String javaName() { return AnovaGLMModel.class.getName(); }

    @Override
    public long progressUnits() {
      return 1;
    }

    public MissingValuesHandling missingValuesHandling() {
      if (_missing_values_handling instanceof MissingValuesHandling)
        return (MissingValuesHandling) _missing_values_handling;
      assert _missing_values_handling instanceof DeepLearningModel.DeepLearningParameters.MissingValuesHandling;
      switch ((DeepLearningModel.DeepLearningParameters.MissingValuesHandling) _missing_values_handling) {
        case MeanImputation:
          return MissingValuesHandling.MeanImputation;
        case Skip:
          return MissingValuesHandling.Skip;
        default:
          throw new IllegalStateException("Unsupported missing values handling value: " + _missing_values_handling);
      }
    }

    public boolean imputeMissing() {
      return missingValuesHandling() == MissingValuesHandling.MeanImputation ||
              missingValuesHandling() == MissingValuesHandling.PlugValues;
    }

    public DataInfo.Imputer makeImputer() {
      if (missingValuesHandling() == MissingValuesHandling.PlugValues) {
        if (_plug_values == null || _plug_values.get() == null) {
          throw new IllegalStateException("Plug values frame needs to be specified when Missing Value Handling = PlugValues.");
        }
        return new GLM.PlugValuesImputer(_plug_values.get());
      } else { // mean/mode imputation and skip (even skip needs an imputer right now! PUBDEV-6809)
        return new DataInfo.MeanImputer();
      }
    }
  }

  public static class AnovaGLMOutput extends Model.Output {
    DataInfo _dinfo;
    public long _training_time_ms;
    double[][] _global_beta;    // coefficients of all models
    final int _nclasses = 1;
    public String[][] _coefficient_names; // coefficient names of all models
    Family _family;
    public String _transformed_columns_key;

    @Override
    public ModelCategory getModelCategory() { return ModelCategory.Regression; }

    public String[][] coefficientNames() { return _coefficient_names; }

    public AnovaGLMOutput(AnovaGLM b, DataInfo dinfo) {
      super(b, dinfo._adaptedFrame);
      _dinfo = dinfo;
      _domains = dinfo._adaptedFrame.domains();
      _family = b._parms._family;
    }
  }

  @Override
  protected Futures remove_impl(Futures fs, boolean cascade) {
    super.remove_impl(fs, cascade);
    return fs;
  }

  @Override
  protected AutoBuffer writeAll_impl(AutoBuffer ab) {
    return super.writeAll_impl(ab);
  }

  @Override
  protected Keyed readAll_impl(AutoBuffer ab, Futures fs) {
    return super.readAll_impl(ab, fs);
  }
}

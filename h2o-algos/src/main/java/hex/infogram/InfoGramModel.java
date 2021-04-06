package hex.infogram;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import hex.Model;
import hex.ModelMetrics;
import hex.deeplearning.DeepLearningModel;
import hex.glm.GLMModel;
import hex.schemas.DRFV3;
import hex.schemas.DeepLearningV3;
import hex.schemas.GBMV3;
import hex.schemas.GLMV3;
import hex.tree.drf.DRFModel;
import hex.tree.gbm.GBMModel;
import water.Job;
import water.Key;
import water.api.schemas3.ModelParametersSchemaV3;

import java.lang.reflect.Field;
import java.util.*;

public class InfoGramModel extends Model<InfoGramModel, InfoGramModel.InfoGramParameter, InfoGramModel.InfoGramOutput> {
  /**
   * Full constructor
   *
   * @param selfKey
   * @param parms
   * @param output
   */
  public InfoGramModel(Key<InfoGramModel> selfKey, InfoGramParameter parms, InfoGramOutput output) {
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

  public static class InfoGramParameter extends Model.Parameters {
    public Algorithm _algorithm = Algorithm.AUTO;     // default to AUTO algorithm which will be GBM
    public String _algorithm_params = new String();   // store user specific parameters for chosen algorithm
    public String[] _sensitive_attributes = null;     // store sensitive features to be excluded from final model
    public double _conditional_info_threshold = 0.1;  // default set by Deep
    public double _varimp_threshold = 0.1;            // default set by Deep
    public double _data_fraction = 1.0;              // fraction of data to use to calculate infogram
    public Model.Parameters _algorithm_parameters;   // store parameters of chosen algorithm
    public int _parallelism = 0;                     // number of models to build in parallel.  if 1, no parallelism
    public int _ntop = 0;                           // if 0 consider all predictors, otherwise, consider topk predictors
    public boolean _pval = false;                   // if true, will calculate p-value
    
    public enum Algorithm {
      AUTO,
      deeplearning,
      drf,
      gbm,
      glm
    }
    
    @Override
    public String algoName() {
      return "INFOGRAM";
    }

    @Override
    public String fullName() {
      return "INFOGRAM";
    }

    @Override
    public String javaName() {
      return InfoGramModel.class.getName();
    }

    @Override
    public long progressUnits() {
      return 0;
    }

    /**
     * This method performs the following functions:
     * 1. extract the algorithm specific parameters from _algorithm_params to _algorithm_parameters which will be 
     * one of GBMParameters, DRFParameters, DeepLearningParameters, GLMParameters.
     * 2. Next, it will copy the parameters that are common to all algorithms from InfoGramParameters to 
     * _algorithm_parameters.
     */
    public void fillImpl() {
      Properties p = new Properties();
      if (_algorithm_params != null && !_algorithm_params.isEmpty()) {
        HashMap<String, String[]> map = new Gson().fromJson(_algorithm_params, new TypeToken<HashMap<String, String[]>>() {
        }.getType());
        for (Map.Entry<String, String[]> param : map.entrySet()) {
          String[] paramVal = param.getValue();
          if (paramVal.length == 1) {
            p.setProperty(param.getKey(), paramVal[0]);
          } else {
            p.setProperty(param.getKey(), Arrays.toString(paramVal));
          }
        }
      }
      
      ModelParametersSchemaV3 paramsSchema;
      Model.Parameters params;
      switch (_algorithm) {
        case glm:
          paramsSchema = new GLMV3.GLMParametersV3();
          params = new GLMModel.GLMParameters();
          // FIXME: This is here because there is no Family.AUTO. It enables us to know if the user specified family or not.
          // FIXME: Family.AUTO will be implemented in https://0xdata.atlassian.net/projects/PUBDEV/issues/PUBDEV-7444
          ((GLMModel.GLMParameters) params)._family = null;
          break;
        case AUTO:
        case gbm:
          paramsSchema = new GBMV3.GBMParametersV3();
          params = new GBMModel.GBMParameters();
          break;
        case drf:
          paramsSchema = new DRFV3.DRFParametersV3();
          params = new DRFModel.DRFParameters();
          break;
        case deeplearning:
          paramsSchema = new DeepLearningV3.DeepLearningParametersV3();
          params = new DeepLearningModel.DeepLearningParameters();
          break;
        default:
          throw new UnsupportedOperationException("Unknown algo: " + _algorithm.name());
      }

      paramsSchema.init_meta();
      _algorithm_parameters = (Model.Parameters) paramsSchema
              .fillFromImpl(params)
              .fillFromParms(p, true)
              .createAndFillImpl();
      copyInfoGramParams(); // copy over InfoGramParameters that are applicable to algorithm_parameters
    }
    
    public void copyInfoGramParams() {
      Field[] algoParams = Parameters.class.getDeclaredFields();
      Field algoField;
      for (Field oneField : algoParams) {
        try {
          algoField = this.getClass().getField(oneField.getName());
          algoField.set(_algorithm_parameters, oneField.get(this));
        } catch (IllegalAccessException | NoSuchFieldException e) { // suppress error printing.  Only care about fields that are accessible
          ;
        }
      }
    }
  }
  
  public static class InfoGramOutput extends Model.Output {
    public double[] _conditional_info;  // conditional info for admissible features in _admissible_features
    public double[] _varimp;  // varimp values for admissible features in _admissible_features
    public String[] _admissible_features; // predictors chosen that exceeds both conditional_info and varimp thresholds
    public InfoGramOutput() { super(); }
    
    public InfoGramOutput(Job job) { _job = job; }
    
    
  }
}

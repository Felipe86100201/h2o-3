package hex.schemas;

import hex.anovaglm.AnovaGLMModel;
import water.api.API;
import water.api.schemas3.ModelOutputSchemaV3;
import water.api.schemas3.ModelSchemaV3;
import water.api.schemas3.TwoDimTableV3;

public class AnovaGLMModelV3 extends ModelSchemaV3<AnovaGLMModel, AnovaGLMModelV3, AnovaGLMModel.AnovaGLMParameters,
        AnovaGLMV3.AnovaGLMParametersV3, AnovaGLMModel.AnovaGLMOutput, AnovaGLMModelV3.AnovaGLMOutputV3> {
  public static final class AnovaGLMOutputV3 extends ModelOutputSchemaV3<AnovaGLMModel.AnovaGLMOutput, AnovaGLMOutputV3> {
    @API(help="Table of Coefficients")
    TwoDimTableV3 coefficients_table;

    @API(help = "GLM model summary")
    TwoDimTableV3 glm_model_summary;

    @API(help="Table of Standardized Coefficients Magnitudes")
    TwoDimTableV3 standardized_coefficient_magnitudes;

    @API(help="Variable Importances", direction=API.Direction.OUTPUT, level = API.Level.secondary)
    TwoDimTableV3 variable_importances;

    @API(help="GLM Z values.  For debugging purposes only")
    double[] glm_zvalues;

    @API(help="GLM p values.  For debugging purposes only")
    double[] glm_pvalues;

    @API(help="GLM standard error values.  For debugging purposes only")
    double[] glm_std_err;
  }
  public AnovaGLMV3.AnovaGLMParametersV3 createParametersSchema() { return new AnovaGLMV3.AnovaGLMParametersV3();}
  public AnovaGLMOutputV3 createOutputSchema() { return new AnovaGLMOutputV3();}

  @Override
  public AnovaGLMModel createImpl() {
    AnovaGLMModel.AnovaGLMParameters parms = parameters.createImpl();
    return new AnovaGLMModel(model_id.key(), parms, null);
  }
}

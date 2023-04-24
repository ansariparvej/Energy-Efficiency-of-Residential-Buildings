from energy_efficiency.constant.training_pipeline import SAVED_MODEL_DIR
from energy_efficiency.ml.model.estimator import ModelResolver
from energy_efficiency.utils.main_utils import load_object
import streamlit as st
import pandas as pd


def load_prediction(input_data):

    # Convert JSON to DataFrame Using read_json()
    df = pd.DataFrame(input_data, index=[0])
    model_resolver = ModelResolver(model_dir=SAVED_MODEL_DIR)
    if not model_resolver.is_model_exists():
        return "Model is not available."
    best_model_path = model_resolver.get_best_model_path()
    model = load_object(file_path=best_model_path)
    prediction = model.predict(df)
    return prediction


def main():

    # title
    st.title(':scroll: **:blue[ENERGY EFFICIENCY]** :scroll:')
    st.text("")
    st.text("")
    st.text("")

    # specifications
    st.markdown(":dart: **:red[_Inputs Given:_]** :dart:")
    st.markdown(":green[Relative Compactness, Surface Area - m², Wall Area - m², Roof Area - m², Overall Height - m, Orientation - 2: North, 3: East, 4: South, 5: West, Glazing Area - 0%, 10%, 25%, 40% (of floor area), Glazing Area Distribution (Variance) - 1: Uniform, 2: North, 3: East, 4: South, 5: West]")
    st.text("")
    st.text("")
    st.markdown(":dart: **:red[_Multi Outputs:_]** :dart:")
    st.markdown(":green[Heating Load - kW, Cooling Load - kW]")
    st.text("")
    st.text("")

    # getting the input data from user
    st.markdown(":dart: **:blue[_Choose Your Requirements for a Residential Building._]** :dart:")

    try:
        st.text("")
        st.markdown(":point_down: **:red[Select the required values from below:]** :point_down:")

        relative_compactness = st.slider("**:blue[Relative Compactness]**", 0.5, 1.0, 0.70)
        surface_area = st.slider("**:blue[Surface Area]**", 500.0, 850.0, 600.0)
        wall_area = st.slider("**:blue[Wall Area]**", 240.0, 420.0, 300.0)
        roof_area = st.slider("**:blue[Roof Area]**", 100.0, 230.0, 150.0)
        overall_height = st.selectbox("**:blue[Overall Height]**", (3.5, 7))
        orientation = st.selectbox("**:blue[Orientation]**", (2, 3, 4, 5))
        glazing_area = st.selectbox("**:blue[Glazing Area]**", (0.0, 0.1, 0.25, 0.40))
        glazing_area_distribution = st.selectbox("**:blue[Glazing Area Distribution]**", (1, 2, 3, 4, 5))
        
        # converting inputs into a json format:

        input_data = {
                      "Relative Compactness": relative_compactness,
                      "Surface Area": surface_area,
                      "Wall Area": wall_area,
                      "Roof Area": roof_area,
                      "Overall Height": overall_height,
                      "Orientation": orientation,
                      "Glazing Area": glazing_area, 
                      "Glazing Area Distribution":glazing_area_distribution
                      }

        if st.button("**:green[Make Prediction]**"):
            prediction = load_prediction(input_data)
            heat = ":sun_with_face:"
            cool = ":rain_cloud:"
            direct = ":point_right:"
            dash = ":wavy_dash:"
            heating_load = f" {heat} Heating Load: {dash} {direct} {prediction[0][0]:0,.3f} kW "
            cooling_load = f" {cool} Cooling Load: {dash} {direct} {prediction[0][1]:0,.3f} kW "
            st.write(heating_load)
            st.write(cooling_load)
    except Exception as e:
        st.text(e)


if __name__ == "__main__":
    main()

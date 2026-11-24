

"""
===============================================================
Librerias 
===============================================================
""" 

import pandas as pd
import numpy as np
import argparse as ap
import os, time, logging, joblib
from pathlib import Path
from copy import deepcopy
# Importar Decision Tree de scikit-learn
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
# Importar objeto para k-fold cross-validation con grid de hyperparámetros
from sklearn.model_selection import GridSearchCV, StratifiedKFold
# Importar objeto para separación de datos en conjuntos de entrenamiento y evaluación
from sklearn.model_selection import train_test_split
# Importar objeto para escalamiento de datos
from sklearn.preprocessing import StandardScaler
# Importar metricas de evaluación
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve
# Importar objeto para imprimir matriz de confusión
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# Importar objeto para imprimir reporte de clasificación
from sklearn.metrics import classification_report
from sklearn.model_selection import ParameterGrid
#imbalance learn para oversampling 
from imblearn.over_sampling import ADASYN, BorderlineSMOTE, KMeansSMOTE, SVMSMOTE
from imblearn.pipeline import Pipeline
from sklearn.cluster import KMeans
# Plots
import matplotlib.pyplot as plt



"""
===============================================================
Argument parser 
===============================================================
""" 

def parser(): 
    """
    Funcion para resivir los arguimentos CLI y guardarlos en el objeto tipo ArgumentParser
    
    Returns: 
        - parser : los argumentos parseados listos para accesar a su valor dentro de este objeto
    """
    parser = ap.ArgumentParser(description="Entrenar y comparar modelo RF, DT y SVM")
    #agregamos todos los argumentos
    parser.add_argument(
        "-p",
        "--prefix",
        type=str,
        help="Id para los resultados",
        default="ML_compar"
    )
    parser.add_argument(
        "-d",
        "--data_s",
        type=str,
        help="Path al data set a utilizar si se proporciona el test (-v) este debe ser el train",
        required=True
    )
    parser.add_argument(
        "-a",
        "--test_data",
        type=str,
        help="Path al data set a de train a utilizar ",
        default=None
    )
    parser.add_argument(
        "-t",
        "--test_porpo",
        type=float,
        help="Proporcion < 1 de datos para el test",
        default=0.2
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        help="Directorio de salida de los resultados",
        default="ML_compar"
    )
    parser.add_argument(
        "-c",
        "--col_inter",
        type=str,
        help="columna de interes para la prediccion",
        default="cardio"
    )
    parser.add_argument(
        "-r",
        "--remove_col",
        type=str,
        help="Lista de id de columnas a eliminar para el training y test del modelo separadas por una coma",
        default=None
    )
    parser.add_argument(
        "-s",
        "--scaled_data",
        help="bandera para especificar si los datos serán escalados",
        action='store_true'
    )
    parser.add_argument(
        "-x",
        "--dt_hp",
        type=str,
        help="Lista de hiperparametros para desición trees separados por | , el hiperparametro con los valores a probar deben ir en el formato Hiper_name:hip_val1,hip_val2,hip_valn | siguiente_hiper_name",
        default=None
    )
    parser.add_argument(
        "-f",
        "--rf_hp",
        type=str,
        help="Lista de hiperparametros para random forest separados por | , el hiperparametro con los valores a probar deben ir en el formato Hiper_name:hip_val1,hip_val2,hip_valn | siguiente_hiper_name",
        default=None
    )
    parser.add_argument(
        "-v",
        "--svm_hp",
        type=str,
        help="Lista de hiperparametros para la SVM separados por | , el hiperparametro con los valores a probar deben ir en el formato Hiper_name:hip_val1,hip_val2,hip_valn | siguiente_hiper_name",
        default=None
    )
    parser.add_argument(
        "-i",
        "--smote_o",
        type=str,
        help="Formas de hacer el SMOTE y cada forma soportada k_means, borderline, SVM deberan ir separados por ; y separados de sus hiperparametros por ^ que cada hiper parametro separado como el de los hiperaprametros de los algortimos ML",
        default=None
    )
    parser.add_argument(
        "-y",
        "--adasyn_o",
        type=str,
        help="Hiper parametros para realizar el ADASYN separados como el de los hiperaprametros de los algortimos ML",
        default=None
    )
    parser.add_argument(
        "-k",
        "--k_folds",
        type=int,
        help="Numero de k folds para la validación cruzada",
        default=5
    )
    parser.add_argument(
        "-j",
        "--save_models",
        help="bandera para especificar si se guardan los modelos",
        action='store_true'
    )
    return parser.parse_args()

"""
===============================================================
Train model 
===============================================================
"""

def _cast_scalar(val):
    """
    Convierte un valor individual desde str al tipo nativo apropiado.
    """
    #si ya esta casteado no se convierte
    if not isinstance(val, str):
        return val  
    
    #limpiar cualquier espacio extra
    clean = val.strip()
    #obtener la clave de cada carcter para tipo binarios o nulos
    lower = clean.lower()

    if lower == "none":
        return None
    if lower == "true":
        return True
    if lower == "false":
        return False

    try:
        return int(clean)
    except ValueError:
        pass
    
    #intentar convertirlo en un dato numerico 
    try:
        return float(clean)
    except ValueError:
        return clean

def parse_hiper(params: dict, prefix: str) -> dict:
    """
    Function that parse the hiperparameters of ML algorithms as well as oversampling ones
    this casts the integers ones as well as it adds the prefix to a pipeline run
    """
    
    parsed_dic = {}
    #recorrer cada hiper parametro con sus valores
    for hp, hp_vals in params.items():
        if isinstance(hp_vals, (str, bytes)) or not hasattr(hp_vals, "__iter__"):
            raise TypeError(
                f"Los valores de '{hp}' deben venir en una lista o iterable, se recibió: {type(hp_vals).__name__}"
            )
        #agregar ek prefijo al hiper parametro si es que no lo tiene
        key = hp if hp.startswith(f"{prefix}__") else f"{prefix}__{hp}"
        #permitir listas con distintos tipos de datos
        parsed_dic[key] = [_cast_scalar(v) for v in hp_vals]

    return parsed_dic

def train_M(model_ML:str, X_train:np.arange, y_train:np.arange, hiper_P:dict=None, k:int=5, name:str="ML_train", over_samp:dict=None, rd:int=42):
    """
    Function that trains 3 different machine learning algorthms 
    Arguments:
        -model_ML: type of model to create, 3 suported options: DT for desicion trees 
            RF for random forest and SVM for suported vector machines 
        -X_train: set of all the variables to be used for training
        -y_train: labels of the variable of interest to be predicted in the X_train 
        -hiper_P: dictionary with the hiperparameter names as keys and a list for the values for 
            the grid for each hiperparameter 
        -name: name to be used in the res df
        -over_samp: dictionary with key as the type of oversampling method (over map s function dic supported values)
        -rd: random state for the run 
    """
    #diccionario para mapear los metodos de oversampling con su funcion 
    over_map={
        "adasyn":[ADASYN(random_state=rd)],
        "k_means":[KMeansSMOTE(random_state=rd)],
        "borderline":[BorderlineSMOTE(random_state=rd)],
        "SVM":[SVMSMOTE(random_state=rd)]
    }
    model_ML_map = {
        "DT": tree.DecisionTreeClassifier(random_state=rd),
        "RF": RandomForestClassifier(random_state=rd),
        "SVM": SVC(random_state=rd),
    }
    #inicializamos los objetos si se va a hacer over samplig o no 
    if over_samp:
        #grid de hiper parametros
        param_grid=[{
                "Osampler":over_map[o_sampler],
                **parse_hiper(over_samp[o_sampler], "Osampler"),
                **parse_hiper(hiper_P,"model")
            } for o_sampler in over_samp.keys()
        ]
        print(param_grid)
        #inicializamos el objeto pipeline 
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("Osampler", BorderlineSMOTE(random_state=rd)),
            ("model",model_ML_map[model_ML]),
        ])
    else:
        #grid the hiper parametros
        param_grid=parse_hiper(hiper_P,"model")
        #inicializamos el objeto pipeline solo con el clasificador
        pipe = Pipeline([
            ("model",model_ML_map[model_ML])
        ])
    #configurar el k-fold cross validation 
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=rd)
    #hacer el grid de hiperparametros 
    models_grid= GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
    )
    grid_df=pd.DataFrame(ParameterGrid(models_grid.param_grid))
    #entrenar el modelo
    models_grid.fit(X_train, y_train)
    #data frame con los mejores resultados
    res_df = pd.DataFrame({
        "Model": [name],
        "hiperparameters": [str(models_grid.best_params_)],
        "F-1_train_cv": [models_grid.best_score_]
    })
    #guardamos el mejor modelo
    model_f=models_grid.best_estimator_
    return [model_f, res_df, grid_df]


"""
===============================================================
Test model 
===============================================================
"""

def make_conf_mat(y_test, y_pred,ax_p=None,conf_mat=None, norm:bool=None, title:str="Matriz de confusión"):
    """
    Function to plot the confusion matrix in absoluote and normalized way
    Arguments: 
        -y_test: true label of the ones that are predicted
        -y_pred: predicted labels 
        -norm: flag to sign if the matrix is gonna be normalized or not 
    """
    if not ax_p:
        conf_mat, ax_p=plt.subplots(figsize=(10, 10), layout="constrained")
    #calculate the confusion matrix 
    confusion_calc= confusion_matrix(y_test, y_pred, normalize=norm)
    conf_disp=ConfusionMatrixDisplay(confusion_matrix=confusion_calc)
    conf_disp.plot(cmap=plt.cm.Blues, ax=ax_p, colorbar=False, text_kw={"fontsize": 14})
    ax_p.set_title(title, fontsize=24)
    ax_p.tick_params(axis="x", labelsize=16)
    ax_p.tick_params(axis="y", labelsize=16)
    ax_p.xaxis.label.set_size(16)
    ax_p.yaxis.label.set_size(16)
    #color bar in the same scale
    cbar1 = conf_mat.colorbar(conf_disp.im_, ax=ax_p, fraction=0.046, pad=0.04)
    cbar1.ax.tick_params(labelsize=14)
    
    return conf_mat

def test_M(model_t, X_test, y_test,type_m="macro", name="ML_test", axes=None, fig=None):
    """
    function to test the model 
    Arguments: 
        -model_t: object with the model to be tested 
        -X_test: set of the variables needed to make the prediction
        -y_test: labels of the predicted data
        -type_m: tyope of the metrics to be used, macro or weighted
    """
    #make the prediction test 
    y_pred=model_t.predict(X_test)
    
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(15, 12))
        
    # Matriz absoluta
    make_conf_mat(y_test, y_pred, 
        ax_p=axes[0],conf_mat=fig, 
        title=f"Confusion matrix {name}"
    )
    # Matriz normalizada 
    make_conf_mat(y_test, y_pred, 
        ax_p=axes[1],conf_mat=fig,
        norm='true', title=f"Confusion matrix normalized {name}"
    )
    
    # Reportes de clasificación 
    report_class = classification_report(y_test, y_pred, output_dict=True)
    report_class_f = classification_report(y_test, y_pred)
    res_df = pd.DataFrame({
        "Model": [name],
        f"F-1_score_{type_m}": [f1_score(y_test, y_pred, average=type_m)],
        f"precision_{type_m}": [precision_score(y_test, y_pred, average=type_m)],
        f"recall_{type_m}": [recall_score(y_test, y_pred, average=type_m)],
        f"AUROC_{type_m}": [roc_auc_score(y_test, y_pred)],
        "F-1_score_weighted": [report_class["weighted avg"]["f1-score"]],
        "precision_weighted": [report_class["weighted avg"]["precision"]],
        "recall_weighted": [report_class["weighted avg"]["recall"]]
    })
    
    return [res_df, report_class_f, fig]

def data_set_report(data,logging,var_inter=None, tag="X"):
    """
    Funcion para hacer el reporte de dimensionalidad de un set de datos
    """
    logging.info(f"Vizualización de {tag}:")
    logging.info(f"\n{data.head()}") 
    #si son los predictores 
    if "X" in tag:
        #dimensionalidad del data set 
        logging.info(f"Dimensionalidad del data set:{data.shape}")
        #revisar si existen columnas no definidas
        for col in data.columns: 
            non_defined_val=len(data[data[col]=="NA"].index) + len(data[data[col]==None].index)
            logging.info(f"NA values in {col}: {non_defined_val}")
    elif "y" in tag:
        if not var_inter:
            var_inter=data.columns[0]
        logging.info(f"Categorías únicas de clasificación:\n{data[var_inter].unique()}")
        logging.info(f"Num de intancias en la categoria {var_inter}:")
        logging.info(f"{data.groupby(var_inter).size()}") 

def parse_hp(hp_str:str):
    """
    Function that parses the hiperparameters of a machine learning model
    """
    dic_hp={"HP":
        {hip.strip().split(":")[0]:[hip_val.strip() 
                for hip_val in hip.strip().split(":")[1].strip().split(",")] 
            for hip in hp_str.strip().split("|")
        }
    }
    return dic_hp


"""
===============================================================
Main function
===============================================================
"""

def main():
    args= parser()
    pref=args.prefix
    out_dir=args.outdir
    #data arguments 
    data_set=args.data_s
    test_set=args.test_data
    remove_cols=args.remove_col.strip().split(",") if args.remove_col else []
    test_propor=args.test_porpo
    var_interes=args.col_inter
    scaled_data=args.scaled_data

    #training agrguments
    k_d=args.k_folds
    #model arguments
    dt_hp={"DT":parse_hp(args.dt_hp)} if args.dt_hp else None
    rf_hp={"RF":parse_hp(args.rf_hp)} if args.rf_hp else None
    svm_hp={"SVM":parse_hp(args.svm_hp)} if args.svm_hp else None
    save_mod=args.save_models
    
    #oversampling arguments
    smote_d={ smote_t.strip().split("^")[0]: parse_hp(smote_t.strip().split("^")[1])["HP"]
        for smote_t in args.smote_o.strip().split(";")
    } if args.smote_o else None
    adasyn_d={"adasyn":parse_hp(args.adasyn_o)["HP"]}if args.adasyn_o else None

    print(smote_d)
    print(adasyn_d)
    #make the each hiperparameters dic
    res_dic={ml_m : hp for dic in [dt_hp,rf_hp,svm_hp]  if dic for ml_m, hp in dic.items()}

    over_dic={over_m : hp for dic in [smote_d, adasyn_d]  if dic for over_m, hp in dic.items()}

    #logging info 
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()
    logging.basicConfig(
        filename=os.path.join(out_dir,f"{pref}_execution.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    #fijar un random state para reproducibilida
    random_state = 42

    logging.info("Leeyendo el el data set")
    data_sets={}
    if test_set:
        data_sets["X_train"]=pd.read_csv(data_set, index_col=0).rename_axis(None)
        data_sets["X_test"]=pd.read_csv(test_set, index_col=0).rename_axis(None)
        #dividir de las etiquetas 
        for data_t in ["X_train", "X_test"]:
            #eliminamos las columnas no deseadas 
            if remove_cols:
                data_sets[data_set].drop(remove_cols, axis=1, inplace=True)
            #obtenemos si es train o test 
            set_t=data_t.split("_")[1]
            #creamos el de las etiquetas
            data_sets[f"y_{set_t}"]=data_sets[data_t].loc[:,[var_interes]]
            #guardamos el data set sin el label
            data_sets[data_t]=data_sets[data_t].loc[:,data_sets[data_t].columns[data_sets[data_t].columns != var_interes]]
    else:
        
        X=pd.read_csv(data_set, index_col=0).rename_axis(None)
        if remove_cols:
            X.drop(remove_cols, axis=1, inplace=True)
        #obtenemos las etiquetas 
        y=X.loc[:,var_interes]
        X=X.loc[:,X.columns[X.columns != var_interes]]
        #split the datasets
        data_sets["X_train"],data_sets["X_test"],data_sets["y_train"],data_sets["y_test"]=train_test_split(X, 
            y, test_size=test_propor, random_state=random_state, stratify=y
        )

    scaler = StandardScaler()
    #Exploración basica de los data sets 
    for data_t in data_sets.keys():
        data_set_report(data_sets[data_t], logging, var_interes, tag=data_t)

    #escalar los datos solo si es necesario
    if scaled_data:
        if not over_dic:
            logging.info("Escalando los datos")
            data_sets["X_train"]=scaler.fit_transform(data_sets["X_train"])
            data_sets["X_test"]=scaler.transform( data_sets["X_test"])
        else:
            logging.info("La ejecución hara sobre sampleo, los datos se escalaran entonces")
    
    #data frame de resultadps
    res_df=pd.DataFrame()
    
    #out paths 
    Path(f"{out_dir}/Test").mkdir(parents=True, exist_ok=True)
    Path(f"{out_dir}/Train").mkdir(parents=True, exist_ok=True)
    Path(f"{out_dir}/models").mkdir(parents=True, exist_ok=True)
    if not res_dic:
        logging.info("No se seleccionaron hiperparametros para ningun modelo")
    #entrenar y evaluar por cada modelo 
    for ML_a in res_dic.keys(): 
        #se hace el entrenamiento
        logging.info(f"{'='*40}\nTraining {ML_a} model {time.perf_counter() - start_time:.2f}s\n{'='*40}")
        train_res=train_M(ML_a,data_sets["X_train"], data_sets["y_train"],
            res_dic[ML_a]["HP"],k=k_d,over_samp=over_dic, name=ML_a
        )
        model_t=train_res[0]
        res_df_tr=train_res[1]
        grid_t=train_res[2]
        logging.info(f"Training {ML_a} finished {time.perf_counter() - start_time:.2f}s\nGrid training:\n{grid_t}")
        #save the grid 
        grid_t.to_csv(os.path.join(out_dir,"Train",f"{pref}_{ML_a}_grid_hip.csv"))
        
        logging.info(f"Testing {ML_a} {time.perf_counter() - start_time:.2f}s")
        #make the tsting
        test_res=test_M(model_t,data_sets["X_test"], 
            data_sets["y_test"], 
            name=ML_a
        )
        res_df_ts = test_res[0]
        logging.info(f"Training finished {ML_a} {time.perf_counter() - start_time:.2f}s, report:\n{test_res[1]}")
        
        model_full_metrics = pd.merge(res_df_tr, res_df_ts, on="Model")
        # Guardamos en el diccionario si quieres mantener ese registro
        res_dic[ML_a]["Res"] = model_full_metrics
        #add to the res df
        res_df = pd.concat([res_df, model_full_metrics], ignore_index=True)
        
        #save the confusion matrix 
        test_res[2].savefig(os.path.join(out_dir,"Test",f"{pref}_{ML_a}_confusion_matrix.png"))
        
        #finally save the model if it is desired 
        if save_mod:
            joblib.dump(model_t, os.path.join(out_dir, "models", f"{pref}_{ML_a}_model.joblib"))
        
    #finally save the results 
    res_df.to_csv(os.path.join(out_dir,f"{pref}_final_compar_res.csv"))  
    
if __name__ == "__main__":
    main()
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit, minimize_scalar
from scipy.integrate import solve_ivp

st.title("Modelo poblacional con vecindades")

# ---------------------------------
# PARÁMETROS GENERALES
# ---------------------------------

n_regiones = st.number_input(
    "Cantidad de regiones",
    min_value=1,
    value=6,
    step=1
)

regiones = [f"R{i+1}" for i in range(n_regiones)]

# ---------------------------------
# VALORES INICIALES
# ---------------------------------

POB_default = {
    "R1": [50, 70, 95, 125],
    "R2": [50, 68, 92, 120],
    "R3": [50, 60, 72, 88],
    "R4": [50, 58, 75, 100],
    "R5": [50, 55, 85, 110],
    "R6": [50, 52, 60, 72]
}

INF_default = {
    "R1": [1, 1, 1, 1],
    "R2": [1, 1, 1, 1],
    "R3": [0.5, 0.5, 0.5, 0.5],
    "R4": [0, 0, 0.5, 0.5],
    "R5": [0, 0, 0, 0],
    "R6": [0, 0, 0, 0]
}

INF_ACT_default = {
    "R1": 1,
    "R2": 1,
    "R3": 0.5,
    "R4": 1,
    "R5": 0,
    "R6": 0
}

Vecinos_default = {
    "R1": ["R2"],
    "R2": ["R1", "R3"],
    "R3": ["R2", "R4"],
    "R4": ["R3", "R5"],
    "R5": ["R4", "R6"],
    "R6": ["R5"]
}

# ---------------------------------
# CARGA DE DATOS
# ---------------------------------

st.header("Datos")

POB = {}
INF = {}
INF_ACT = {}
Vecinos = {}

for r in regiones:

    st.subheader(r)

    pob_default = ",".join(
        map(str, POB_default.get(r, [50,60,70,80]))
    )

    inf_default = ",".join(
        map(str, INF_default.get(r, [0,0,0,0]))
    )

    vecinos_default = ",".join(
        Vecinos_default.get(r, [])
    )

    poblaciones_txt = st.text_input(
        f"Poblaciones históricas de {r}",
        value=pob_default,
        key=f"pob_{r}"
    )

    infra_txt = st.text_input(
        f"Infraestructuras históricas de {r}",
        value=inf_default,
        key=f"inf_{r}"
    )

    infra_act = st.number_input(
        f"Infraestructura actual de {r}",
        value=float(
            INF_ACT_default.get(r, 0)
        ),
        key=f"inf_act_{r}"
    )

    vecinos_txt = st.text_input(
        f"Vecinos de {r}",
        value=vecinos_default,
        key=f"vec_{r}"
    )

    POB[r] = [
        float(x)
        for x in poblaciones_txt.split(",")
    ]

    INF[r] = [
        float(x)
        for x in infra_txt.split(",")
    ]

    INF_ACT[r] = infra_act

    Vecinos[r] = [
        x.strip()
        for x in vecinos_txt.split(",")
        if x.strip() != ""
    ]

# ---------------------------------
# BOTÓN PRINCIPAL
# ---------------------------------

if st.button("Ejecutar modelo"):

    T_hist = len(next(iter(POB.values()))) - 1

    T_f = T_hist + 1

    # ---------------------------------
    # AJUSTE DE PARÁMETROS
    # ---------------------------------

    def calcular_r0_alfa(POB, INF, regiones):

        parametros = {}

        for r in regiones:

            P = np.array(POB[r])

            I = np.array(INF[r])

            r_obs = np.log(P[1:] / P[:-1])

            I_mid = (I[1:] + I[:-1]) / 2

            def modelo_lineal(I, r0, alfa):

                return r0 + alfa * I

            r0, alfa = curve_fit(
                modelo_lineal,
                I_mid,
                r_obs
            )[0]

            parametros[r] = (r0, alfa)

        return parametros

    parametros = calcular_r0_alfa(
        POB,
        INF,
        regiones
    )

    # ---------------------------------
    # INFRAESTRUCTURA
    # ---------------------------------

    def infraestructura(region, t):

        I_hist = INF[region]

        I_total = I_hist + [INF_ACT[region]]

        tiempos = np.arange(len(I_total))

        return np.interp(
            t,
            tiempos,
            I_total
        )

    # ---------------------------------
    # SISTEMA
    # ---------------------------------

    def sistema(t, Pvec, beta):

        dP = np.zeros(len(regiones))

        for i, r in enumerate(regiones):

            r0, alfa = parametros[r]

            Ii = infraestructura(r, t)

            crecimiento = (
                r0 + alfa * Ii
            ) * Pvec[i]

            difusion = 0

            for vecino in Vecinos[r]:

                if vecino in regiones:

                    j = regiones.index(vecino)

                    difusion += (
                        Pvec[j] - Pvec[i]
                    )

            difusion *= beta

            dP[i] = crecimiento + difusion

        return dP

    # ---------------------------------
    # CONDICIÓN INICIAL
    # ---------------------------------

    P0_hist = [
        POB[r][0]
        for r in regiones
    ]

    # ---------------------------------
    # ERROR(beta)
    # ---------------------------------

    def error_beta(beta):

        tiempos_censo = np.arange(
            T_hist + 1
        )

        sol = solve_ivp(
            sistema,
            [0, T_hist],
            P0_hist,
            args=(beta,),
            t_eval=tiempos_censo
        )

        error = 0

        for i, r in enumerate(regiones):

            reales = np.array(POB[r])

            modelo = sol.y[i]

            error += np.sum(
                (modelo - reales)**2
            )

        return error

    # ---------------------------------
    # CALIBRACIÓN beta
    # ---------------------------------

    resultado = minimize_scalar(
        error_beta,
        bounds=(0, 1),
        method='bounded'
    )

    beta_opt = resultado.x

    st.subheader("Beta óptimo")

    st.write(beta_opt)

    # ---------------------------------
    # RESOLUCIONES SUCESIVAS
    # ---------------------------------

    resultados = []

    for i0 in range(T_hist + 1):

        P0 = [
            POB[r][i0]
            for r in regiones
        ]

        sol = solve_ivp(
            sistema,
            [i0, T_f],
            P0,
            args=(beta_opt,),
            t_eval=np.linspace(
                i0,
                T_f,
                300
            )
        )

        fila = {
            "inicio": i0
        }

        for j, r in enumerate(regiones):

            fila[r] = sol.y[j][-1]

        resultados.append(fila)

    sol_final = sol

    # ---------------------------------
    # TABLA
    # ---------------------------------

    st.subheader("Predicciones")

    df = pd.DataFrame(resultados)

    st.dataframe(df)

    # ---------------------------------
    # GRÁFICO
    # ---------------------------------

    st.subheader("Proyección final")

    fig, ax = plt.subplots()

    for j, r in enumerate(regiones):

        ax.plot(
            sol_final.t,
            sol_final.y[j],
            label=r
        )

        ax.scatter(
            np.arange(len(POB[r])),
            POB[r]
        )

    ax.grid()

    ax.set_xlabel("Tiempo")

    ax.set_ylabel("Población")

    ax.legend()

    st.pyplot(fig)

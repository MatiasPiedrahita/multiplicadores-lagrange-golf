"""
Simulador de ganancias - Pro-T (pelotas de golf)
Aplicación pensada para usuarios no técnicos (finanzas, gerencia).
El cálculo se hace con multiplicadores de Lagrange usando SymPy.
"""
import numpy as np
import sympy as sp
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Simulador de ganancias · Pro-T", page_icon="⛳", layout="wide")

# Valores originales del ejercicio
BASE = dict(presupuesto=216.0, costo_x=20.0, costo_y=4.0,
            a=48.0, b=96.0, c=1.0, d=2.0, e=9.0)


# ======================================================================
# 1. MOTOR MATEMÁTICO (SymPy): se resuelve UNA vez con parámetros
#    simbólicos y luego solo se reemplazan los números.
# ======================================================================
@st.cache_resource
def motor():
    x, y, lam = sp.symbols("x y lambda", real=True)
    a, b, c, d, e, p, q, P = sp.symbols("a b c d e p q P", real=True)

    f = a*x + b*y - c*x**2 - d*x*y - e*y**2      # ganancia
    g = p*x + q*y                                 # gasto
    L = f - lam*(g - P)                           # lagrangiano

    ecuaciones = [sp.diff(L, x), sp.diff(L, y), sp.diff(L, lam)]
    sol = sp.solve(ecuaciones, [x, y, lam], dict=True)[0]

    # Máximo sin restricción de presupuesto (gradiente = 0)
    sol_libre = sp.solve([sp.diff(f, x), sp.diff(f, y)], [x, y], dict=True)[0]

    prm = (a, b, c, d, e, p, q, P)
    return dict(
        lagrange=sp.lambdify(prm, [sol[x], sol[y], sol[lam]], "math"),
        libre=sp.lambdify((a, b, c, d, e), [sol_libre[x], sol_libre[y]], "math"),
    )


def ganancia(prm, x, y):
    a, b, c, d, e = prm[:5]
    return a*x + b*y - c*x**2 - d*x*y - e*y**2


def gradiente(prm, x, y):
    a, b, c, d, e = prm[:5]
    return np.array([a - 2*c*x - d*y, b - d*x - 2*e*y])


def optimizar(prm):
    """Mejor combinación (x, y) que gasta todo el presupuesto, sin cantidades negativas."""
    a, b, c, d, e, p, q, P = prm
    candidatos = []
    try:
        xs, ys, ls = motor()["lagrange"](*prm)
        if xs >= -1e-9 and ys >= -1e-9:
            candidatos.append(("interior", max(xs, 0.0), max(ys, 0.0)))
    except (ZeroDivisionError, ValueError, TypeError):
        pass
    # Extremos de la recta de presupuesto (todo a un solo rubro)
    candidatos.append(("solo_produccion", P/p, 0.0))
    candidatos.append(("solo_publicidad", 0.0, P/q))
    tipo, xs, ys = max(candidatos, key=lambda k: ganancia(prm, k[1], k[2]))
    return tipo, xs, ys, ganancia(prm, xs, ys)


def valor_marginal(prm):
    """Cuánto sube la ganancia máxima por cada unidad extra de presupuesto
    (en el caso interior coincide exactamente con lambda)."""
    P = prm[-1]
    h = max(P*1e-4, 1e-6)
    h = min(h, P/2)
    z_mas = optimizar(prm[:-1] + (P + h,))[3]
    z_menos = optimizar(prm[:-1] + (P - h,))[3]
    return (z_mas - z_menos) / (2*h)


def optimo_libre(prm):
    """Máximo de la ganancia si no hubiera límite de presupuesto (si existe)."""
    a, b, c, d, e = prm[:5]
    if not (c > 0 and 4*c*e - d**2 > 0):          # la función debe ser cóncava
        return None
    xu, yu = motor()["libre"](a, b, c, d, e)
    if xu < 0 or yu < 0:
        return None
    return xu, yu, ganancia(prm, xu, yu)


# ======================================================================
# 2. FORMATO "humano" de cifras
# ======================================================================
def dinero(miles):
    v = miles * 1000
    s = f"${abs(v):,.0f}".replace(",", ".")
    return "-" + s if v < 0 else s


def num(v, dec=0):
    s = f"{v:,.{dec}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


# ======================================================================
# 3. ENTRADAS (barra lateral)
# ======================================================================
for k, v in BASE.items():
    st.session_state.setdefault(k, v)


def restablecer():
    for k, v in BASE.items():
        st.session_state[k] = v


with st.sidebar:
    st.header("⚙️ Datos del escenario")
    st.caption("Cambie los valores y los resultados se actualizan al instante. "
               "Todas las cifras de dinero están en **miles de dólares**.")

    st.slider("💵 Presupuesto mensual (miles de USD)", 10.0, 800.0, step=1.0, key="presupuesto",
              help="Dinero disponible al mes para producir pelotas y pagar publicidad.")
    st.number_input("🏭 Costo de producir 1.000 pelotas (miles de USD)", min_value=0.5,
                    max_value=200.0, step=0.5, key="costo_x")
    st.number_input("📣 Costo de 1 hora de publicidad (miles de USD)", min_value=0.5,
                    max_value=200.0, step=0.5, key="costo_y")

    with st.expander("📈 Modelo de ganancia (opcional, avanzado)"):
        st.caption("Ganancia = a·x + b·y − c·x² − d·x·y − e·y²  \n"
                   "x = miles de pelotas, y = horas de publicidad")
        c1, c2 = st.columns(2)
        c1.number_input("a", step=1.0, key="a", help="Aporte de cada mil pelotas vendidas")
        c2.number_input("b", step=1.0, key="b", help="Aporte de cada hora de publicidad")
        c1.number_input("c", min_value=0.0, step=0.5, key="c", help="Saturación por producir mucho")
        c2.number_input("d", min_value=0.0, step=0.5, key="d", help="Efecto cruzado producción-publicidad")
        st.number_input("e", min_value=0.0, step=0.5, key="e", help="Saturación por exceso de publicidad")

    st.button("↩️ Volver a los valores del ejercicio", on_click=restablecer, width="stretch")

S = st.session_state
prm = (S.a, S.b, S.c, S.d, S.e, S.costo_x, S.costo_y, S.presupuesto)
prm_base = tuple(BASE[k] for k in ["a", "b", "c", "d", "e", "costo_x", "costo_y", "presupuesto"])

# ======================================================================
# 4. CÁLCULO
# ======================================================================
tipo, xo, yo, zo = optimizar(prm)
lam = valor_marginal(prm)
_, xb, yb, zb = optimizar(prm_base)
libre = optimo_libre(prm)
presupuesto_ideal = (S.costo_x*libre[0] + S.costo_y*libre[1]) if libre else None

# ======================================================================
# 5. RESULTADOS PRINCIPALES
# ======================================================================
st.title("⛳ Simulador de ganancias · Pro-T")
st.markdown("¿Cuántas pelotas producir y cuántas horas de publicidad contratar para "
            "**ganar lo máximo posible** sin pasarse del presupuesto?")


def delta(actual, base, fmt):
    return None if abs(actual - base) < 1e-9 else fmt(actual - base) + " vs. ejercicio"


m1, m2, m3, m4 = st.columns(4)
m1.metric("🏭 Producción recomendada", f"{num(xo*1000)} pelotas",
          delta(xo*1000, xb*1000, lambda v: ("+" if v > 0 else "") + num(v)))
m2.metric("📣 Publicidad recomendada", f"{num(yo, 1)} horas",
          delta(yo, yb, lambda v: ("+" if v > 0 else "") + num(v, 1)))
m3.metric("💰 Ganancia máxima", dinero(zo),
          delta(zo, zb, lambda v: ("+" if v > 0 else "") + dinero(v)))
m4.metric("➕ Valor de $1 extra de presupuesto", ("-" if lam < 0 else "") + f"${num(abs(lam), 2)}",
          help="Cuánto aumentaría la ganancia por cada dólar adicional de presupuesto. "
               "En matemáticas se llama multiplicador de Lagrange (λ).")

gasto_x, gasto_y = S.costo_x*xo, S.costo_y*yo
resumen = (f"Con un presupuesto de **{dinero(S.presupuesto)}**, lo mejor es producir "
           f"**{num(xo*1000)} pelotas** ({dinero(gasto_x)}) y contratar **{num(yo, 1)} horas "
           f"de publicidad** ({dinero(gasto_y)}). Así la ganancia esperada es **{dinero(zo)}** al mes.")
if lam >= 0:
    resumen += (f" Si el presupuesto subiera {dinero(1)}, la ganancia aumentaría cerca de "
                f"**{dinero(lam)}**.")
st.success(resumen)

if tipo == "solo_produccion":
    st.info("📌 En este escenario conviene destinar **todo** el presupuesto a producción: "
            "la publicidad no compensa su costo.")
elif tipo == "solo_publicidad":
    st.info("📌 En este escenario conviene destinar **todo** el presupuesto a publicidad.")

if libre:
    if lam < 0:
        st.warning(f"⚠️ El presupuesto es mayor de lo necesario: gastarlo todo **reduce** la ganancia. "
                   f"Lo ideal sería gastar solo **{dinero(presupuesto_ideal)}**, con lo que se "
                   f"ganaría **{dinero(libre[2])}**.")
    else:
        st.caption(f"ℹ️ Mientras el presupuesto sea menor a {dinero(presupuesto_ideal)}, cada dólar "
                   f"adicional sigue sumando ganancia. Por encima de esa cifra ya no conviene gastar más; "
                   f"el techo de ganancia del modelo es {dinero(libre[2])}.")
else:
    st.warning("⚠️ Con los coeficientes del modelo actual la ganancia no tiene un techo natural; "
               "revise los valores del modelo avanzado.")

# ======================================================================
# 6. GRÁFICOS
# ======================================================================
x_max = 1.4 * max(xo, S.presupuesto/S.costo_x, libre[0] if libre else 0, 1)
y_max = 1.6 * max(yo, libre[1] if libre else 0, 1)
gx = np.linspace(0, x_max, 160)
gy = np.linspace(0, y_max, 160)
X, Y = np.meshgrid(gx, gy)
Z = ganancia(prm, X, Y)

lx = np.linspace(0, S.presupuesto/S.costo_x, 200)
ly = (S.presupuesto - S.costo_x*lx) / S.costo_y
dentro = (ly >= 0) & (ly <= y_max)
lx, ly = lx[dentro], ly[dentro]

tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Mapa de ganancias", "🧊 Vista 3D",
                                  "💵 ¿Y si cambia el presupuesto?", "🧮 Cómo se calcula"])

with tab1:
    st.caption("Cada curva une combinaciones con la **misma ganancia** (como las curvas de nivel de un mapa). "
               "La línea roja muestra todo lo que se puede comprar gastando el presupuesto completo. "
               "El punto óptimo ⭐ es donde la línea roja apenas **toca** la curva de ganancia más alta.")
    fig = go.Figure()
    fig.add_trace(go.Contour(x=gx, y=gy, z=Z, colorscale="Tealgrn", ncontours=22,
                             contours=dict(showlabels=True, labelfont=dict(size=10, color="black")),
                             colorbar=dict(title="Ganancia<br>(miles USD)"),
                             hovertemplate="Pelotas: %{x:.1f} mil<br>Publicidad: %{y:.1f} h"
                                           "<br>Ganancia: %{z:.0f} mil USD<extra></extra>"))
    fig.add_trace(go.Contour(x=gx, y=gy, z=Z, showscale=False, hoverinfo="skip",
                             contours=dict(start=zo, end=zo, size=1, coloring="lines"),
                             line=dict(color="#1d3557", width=3, dash="dash"),
                             name="Curva de la ganancia máxima", showlegend=True))
    fig.add_trace(go.Scatter(x=lx, y=ly, mode="lines", line=dict(color="#e63946", width=4),
                             name="Presupuesto gastado completo"))
    fig.add_trace(go.Scatter(x=[xo], y=[yo], mode="markers+text", text=[f"  Óptimo: {dinero(zo)}"],
                             textposition="middle right", textfont=dict(size=13, color="black"),
                             marker=dict(symbol="star", size=20, color="gold",
                                         line=dict(color="black", width=1.5)),
                             name="Mejor combinación"))
    if libre:
        fig.add_trace(go.Scatter(x=[libre[0]], y=[libre[1]], mode="markers",
                                 marker=dict(symbol="x", size=12, color="#1d3557"),
                                 name="Techo de ganancia (sin límite de presupuesto)"))
    # Flecha: dirección en que más crece la ganancia (gradiente)
    gr = gradiente(prm, xo, yo)
    if np.linalg.norm(gr) > 1e-9:
        s = 0.18 / max(abs(gr[0])/x_max, abs(gr[1])/y_max)
        fig.add_annotation(x=xo + s*gr[0], y=yo + s*gr[1], ax=xo, ay=yo, xref="x", yref="y",
                           axref="x", ayref="y", showarrow=True, arrowhead=3, arrowsize=1.4,
                           arrowwidth=2.5, arrowcolor="#1d3557",
                           text="Hacia donde más<br>crece la ganancia", font=dict(size=11))
    fig.update_layout(height=560, xaxis_title="Pelotas producidas (miles al mes)",
                      yaxis_title="Publicidad (horas al mes)", margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=-0.18))
    fig.update_xaxes(range=[0, x_max])
    fig.update_yaxes(range=[0, y_max])
    st.plotly_chart(fig, width="stretch")

with tab2:
    st.caption("La 'montaña' es la ganancia. La curva roja es el camino que permite el presupuesto; "
               "la estrella es el punto más alto de ese camino. Puede girar el gráfico con el mouse o el dedo.")
    z_min = float(np.percentile(Z, 2))
    Zc = np.clip(Z, z_min, None)
    fig3 = go.Figure()
    fig3.add_trace(go.Surface(x=gx, y=gy, z=Zc, colorscale="Tealgrn", opacity=0.9,
                              colorbar=dict(title="Ganancia"),
                              contours=dict(z=dict(show=True, usecolormap=True, project_z=True,
                                                   start=z_min, end=float(Z.max()),
                                                   size=(float(Z.max()) - z_min)/18)),
                              hovertemplate="Pelotas: %{x:.1f} mil<br>Publicidad: %{y:.1f} h"
                                            "<br>Ganancia: %{z:.0f}<extra></extra>"))
    fig3.add_trace(go.Scatter3d(x=lx, y=ly, z=ganancia(prm, lx, ly), mode="lines",
                                line=dict(color="#e63946", width=8), name="Presupuesto completo"))
    fig3.add_trace(go.Scatter3d(x=[xo], y=[yo], z=[zo], mode="markers",
                                marker=dict(size=8, color="gold", line=dict(color="black", width=2)),
                                name="Mejor combinación"))
    fig3.update_layout(height=620, margin=dict(l=0, r=0, t=10, b=0),
                       scene=dict(xaxis_title="Pelotas (miles)", yaxis_title="Publicidad (h)",
                                  zaxis_title="Ganancia (miles USD)", aspectmode="cube"),
                       legend=dict(orientation="h", y=0))
    st.plotly_chart(fig3, width="stretch")

with tab3:
    st.caption("¿Cuánto se puede ganar con distintos presupuestos? (manteniendo los costos y el modelo actuales)")
    tope = max(2.5*S.presupuesto, 1.3*presupuesto_ideal if presupuesto_ideal else 0)
    budgets = np.linspace(tope/100, tope, 200)
    zs = [optimizar(prm[:-1] + (B,))[3] for B in budgets]
    fs = go.Figure()
    fs.add_trace(go.Scatter(x=budgets*1000, y=np.array(zs)*1000, mode="lines",
                            line=dict(color="#2a9d8f", width=3), name="Ganancia máxima",
                            hovertemplate="Presupuesto: $%{x:,.0f}<br>Ganancia: $%{y:,.0f}<extra></extra>"))
    fs.add_trace(go.Scatter(x=[S.presupuesto*1000], y=[zo*1000], mode="markers",
                            marker=dict(size=14, color="gold", symbol="star", line=dict(color="black", width=1)),
                            name="Escenario actual"))
    if presupuesto_ideal:
        fs.add_vline(x=presupuesto_ideal*1000, line_dash="dash", line_color="#1d3557",
                     annotation_text=f"Presupuesto ideal: {dinero(presupuesto_ideal)}",
                     annotation_position="top left")
    fs.update_layout(height=450, xaxis_title="Presupuesto mensual (USD)", yaxis_title="Ganancia máxima (USD)",
                     margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fs, width="stretch")
    if presupuesto_ideal:
        st.markdown(f"La ganancia crece con el presupuesto hasta **{dinero(presupuesto_ideal)}**. "
                    f"A partir de ahí, gastar más empieza a restar ganancia (el mercado se satura).")

with tab4:
    st.markdown("Para quien quiera ver el detalle: el problema se resuelve con **multiplicadores de Lagrange**.")
    xs_, ys_, ls_ = sp.symbols("x y lambda")
    R = lambda v: sp.nsimplify(v, rational=True)
    a, b, c, d, e, p, q, P = [R(v) for v in prm]
    f = a*xs_ + b*ys_ - c*xs_**2 - d*xs_*ys_ - e*ys_**2
    g = p*xs_ + q*ys_
    L = f - ls_*(g - P)
    st.markdown("**1. Función de ganancia y restricción de presupuesto**")
    st.latex(r"f(x,y) = " + sp.latex(f) + r"\qquad\text{sujeto a}\qquad " + sp.latex(g) + "=" + sp.latex(P))
    st.markdown("**2. Lagrangiano**")
    st.latex(r"\mathcal{L}(x,y,\lambda) = " + sp.latex(sp.expand(L)))
    st.markdown("**3. Derivadas parciales igualadas a cero**")
    eqs = [sp.diff(L, v) for v in (xs_, ys_, ls_)]
    for v, eq in zip(("x", "y", r"\lambda"), eqs):
        st.latex(r"\frac{\partial \mathcal{L}}{\partial " + v + "} = " + sp.latex(sp.expand(eq)) + " = 0")
    sol = sp.solve(eqs, [xs_, ys_, ls_], dict=True)
    st.markdown("**4. Solución del sistema**")
    if sol:
        s0 = sol[0]
        st.latex(rf"x = {sp.latex(s0[xs_])}\approx {float(s0[xs_]):.3f},\quad "
                 rf"y = {sp.latex(s0[ys_])}\approx {float(s0[ys_]):.3f},\quad "
                 rf"\lambda = {sp.latex(s0[ls_])}\approx {float(s0[ls_]):.3f}")
        st.latex(rf"f(x^*,y^*) \approx {float(f.subs(s0)):.3f}")
        if tipo != "interior":
            st.info("La solución del sistema da una cantidad negativa o no es la mejor, así que el óptimo "
                    "real está en un extremo de la recta de presupuesto (todo a un solo rubro).")
    else:
        st.info("El sistema no tiene solución única con estos valores; se usan los extremos del presupuesto.")

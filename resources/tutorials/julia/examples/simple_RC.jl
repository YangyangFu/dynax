using ModelingToolkitStandardLibrary.Thermal, ModelingToolkit, DifferentialEquations, Plots, Plots.Measures
using .Threads 
using ModelingToolkitStandardLibrary.Blocks
println(Threads.nthreads())

@parameters t

Cz = 6953.9422092947289
Cwe = 21567.368048437285
Cwi = 188064.81655062342
Re = 1.4999822619982095
Ri = 0.55089086571081913
Rw = 5.6456475609117183
Rg = 3.9933826145529263

@named cap_z = HeatCapacitor(C=Cz, T_start=273.15+20)
@named cap_we = HeatCapacitor(C=Cwe, T_start=273.15+32)
@named cap_wi = HeatCapacitor(C=Cwi, T_start=273.15+26.5)
@named res_e = ThermalResistor(R=Re)
@named res_i = ThermalResistor(R=Ri)
@named res_w = ThermalResistor(R=Rw)
@named res_g = ThermalResistor(R=Rg)
@named q_sol_e = PrescribedHeatFlow()
@named q_rad_i = PrescribedHeatFlow()
@named q_conv_i = PrescribedHeatFlow()
@named q_hvac = PrescribedHeatFlow()
@named Tout = PrescribedTemperature()
@named Tz = TemperatureSensor()
@named Twe = TemperatureSensor()
@named Twi = TemperatureSensor()
@named cons30 = Constant(k=273.15+30)
@named cons0 = Constant(k=0)

connections = [
    connect(cons30.output, Tout.T),
    connect(Tout.port, res_g.port_a),
    connect(Tout.port, res_e.port_a),
    connect(res_g.port_b, cap_z.port),
    connect(res_e.port_b, cap_we.port),
    connect(cap_we.port, res_w.port_a),
    connect(res_w.port_b, cap_wi.port),
    connect(cap_wi.port, res_i.port_a),
    connect(res_i.port_b, cap_z.port),
    connect(cons0.output, q_sol_e.Q_flow),
    connect(cons0.output, q_rad_i.Q_flow),
    connect(cons0.output, q_conv_i.Q_flow),
    connect(cons0.output, q_hvac.Q_flow),
    connect(q_sol_e.port, cap_we.port),
    connect(q_rad_i.port, cap_wi.port),
    connect(q_conv_i.port, cap_z.port),
    connect(q_hvac.port, cap_z.port),
    connect(Tz.port, cap_z.port),
    connect(Twe.port, cap_we.port),
    connect(Twi.port, cap_wi.port)
]

@named model = ODESystem(connections, t, 
                        systems=[cap_z, cap_we, cap_wi,
                                res_e, res_i, res_w, res_g,
                                q_sol_e, q_rad_i, q_conv_i, q_hvac,
                                Tout, Tz, Twe, Twi, cons30, cons0])
@time sys = structural_simplify(model)
@time prob = ODEProblem(sys, Pair[], (0, 3600.0*24))

# sovle the problem with Tsit5 sovler for non-stiff system
@time sol = solve(prob, Tsit5())

plot(title = "RC Model")
plot!(sol, vars = [Tz.T, Twe.T, Twi.T], labels = ["Tz" "Twe" "Twi"], margin = 10mm)
plot!(size=(1200,600))
savefig("RC.png")
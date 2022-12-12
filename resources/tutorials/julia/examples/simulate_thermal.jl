using ModelingToolkitStandardLibrary.Thermal, ModelingToolkit, DifferentialEquations, Plots
#using Pkg; Pkg.add("Threads")
using .Threads 

println(Threads.nthreads())

@parameters t

C1 = 15
C2 = 15
@named mass1 = HeatCapacitor(C=C1, T_start=373.15)
@named mass2 = HeatCapacitor(C=C2, T_start=273.15)
@named conduction = ThermalConductor(G=10)
@named Tsensor1 = TemperatureSensor()
@named Tsensor2 = TemperatureSensor()

connections = [
    connect(mass1.port, conduction.port_a),
    connect(conduction.port_b, mass2.port),
    connect(mass1.port, Tsensor1.port),
    connect(mass2.port, Tsensor2.port),
]

@named model = ODESystem(connections, t, systems=[mass1, mass2, conduction, Tsensor1, Tsensor2])
@time sys = structural_simplify(model)
@time prob = ODEProblem(sys, Pair[], (0, 50.0))

# sovle the problem with Tsit5 sovler for non-stiff system
#solver=Tsit5()
solver=ImplicitEuler()
@time sol = solve(prob, solver, dt=0.1)
T_final_K = sol[(mass1.T * C1 + mass2.T * C2) / (C1 + C2)]

plot(title = "Thermal Conduction Demonstration")
plot!(sol, vars = [mass1.T, mass2.T], labels = ["Mass 1 Temperature" "Mass 2 Temperature"])
plot!(sol.t, T_final_K, label = "Steady-State Temperature")
savefig("thermal-simulation.png")
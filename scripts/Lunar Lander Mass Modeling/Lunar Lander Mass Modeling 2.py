# Camden Hill
# Lunar Module Capstone Mass Modeling Simulation
import numpy as np
import matplotlib.pyplot as plt
from sympy import *

def newton_raphson(f,f_,guess,error): # equations should be strings
#    print("The Newton-Raphson Method:")
   value = guess
   i = 0
   while True:
    #  print("Iteration {}: Value = {}, Error = {}".format(i,value,eval(f)))
     i += 1
     value -= eval(f) / eval(f_)
     if np.absolute(eval(f)) < error:
    #    print("Iteration {}: Value = {}, Error = {}".format(i,value,eval(f)))
       return value,i
     if i >= 250:
       return False,value,i

def toroid_raphson_setup(guess,V,rt,max_error): # mean anomaly and eccentricity 
    r = symbols("r")
    f = V - ((2*(np.pi**2)*(r**2)*(rt-r)))
    f_ = diff(f,r)
    value = symbols("value")
    f,f_ = [f.subs(r,value),f_.subs(r,value)] # f_str,fprime_str = [str(f),str(f_)]
    f,f_ = [str(f),str(f_)]
    def tor_raph(f,f_,guess,error=max_error):
        return newton_raphson(f,f_,guess,error)
    return tor_raph,f,f_

def cylinder_raphson_setup(guess,V,l,max_error): # mean anomaly and eccentricity 
    r = symbols("r")
    f = V - (np.pi*((l*(r**2)) - ((2/3)*(r**3))))
    f_ = diff(f,r)
    value = symbols("value")
    f,f_ = [f.subs(r,value),f_.subs(r,value)] # f_str,fprime_str = [str(f),str(f_)]
    f,f_ = [str(f),str(f_)]
    def cyl_raph(f,f_,guess,error=max_error):
        return newton_raphson(f,f_,guess,error)
    return cyl_raph,f,f_

def propellant_mass_calculation(Delta_v,dry_mass,Isp):
    return dry_mass * ((np.exp(Delta_v/(Isp*9.80665)))-1) # m_unloaded * (1-(np.exp(-(Delta_v/(Isp*g_earth)))))

def propellant_volume_calculation(propellant_mass,mixture_ratio,oxidizer_density=1304.03,fuel_density=450.2249): # oxidizer_mass=31.998 , fuel_mass=16.04236,
    oxidizer_total_mass = propellant_mass * (mixture_ratio/(1+mixture_ratio))
    fuel_total_mass = propellant_mass * (1/(1+mixture_ratio))
    return oxidizer_total_mass / oxidizer_density, fuel_total_mass / fuel_density

def sph_tank_calc(liquid_volume,aluminum_thickness,carbon_thickness):
    # liquid_volume = fluid_mass / fluid_density
    liquid_radius = (liquid_volume/((4/3)*np.pi))**(1/3)
    total_radius = liquid_radius + aluminum_thickness + carbon_thickness
    aluminum_volume = ((4/3)*np.pi*((liquid_radius + aluminum_thickness)**3)) - (liquid_volume)
    carbon_volume = ((4/3)*np.pi*((total_radius)**3)) - (liquid_volume + aluminum_volume)
    # total_volume = ((4/3)*np.pi*((total_radius)**3))
    aluminum_mass = aluminum_volume * aluminum_density
    carbon_mass = carbon_volume * carbon_density
    # print("Volume = {} m^3,Radius = {} m, Diameter = {} m, Single Tank Mass = {} kg".format(total_volume,total_radius,total_radius*2,tank_mass))
    return liquid_radius, liquid_radius+aluminum_thickness, liquid_radius+aluminum_thickness+carbon_thickness, aluminum_mass, carbon_mass

def tor_tank_calc(volume,outermost_surface_radius,aluminum_thickness,carbon_thickness,tor_guess,max_error):
    # V=(πr^2)(2πR) # r = circle radius, R = central axis to center circle radius
    # R + r = outermost_surface_radius - carbon_thickness - aluminum_thickness # outermost_surface_radius = R + r + carbon_thickness + aluminum_thickness
    # Completing the cubic
    # a,b,c,d = [1,-(outermost_surface_radius-carbon_thickness-aluminum_thickness),0,volume/(2*(np.pi**2))]
    # term1 = - ((b**3)/(27*(a**3))) + ((b*c)/(6*(a**2))) - (d/(2*a))
    # term2 = (c/(3*a)) - ((b**2)/(9*(a**2)))
    # r = ((term1 - (((term1**2)+(term2**3))**0.5))**(1/3)) + ((term1 + (((term1**2)+(term2**3))**0.5))**(1/3)) - (b/(3*a))
    tor_raph,f,f_ = toroid_raphson_setup(tor_guess,volume,outermost_surface_radius-aluminum_thickness-carbon_thickness,max_error)
    r,iterations = tor_raph(f,f_,tor_guess)
    R = outermost_surface_radius-r-carbon_thickness-aluminum_thickness
    Va = (2*(np.pi**2)*R*((r+aluminum_thickness)**2)) - volume
    if Va < (10**-10):
       Va = 0
    Vc = (2*(np.pi**2)*R*((r+aluminum_thickness+carbon_thickness)**2)) - volume - Va
    if Vc < (10**-10):
       Vc = 0
    aluminum_mass,carbon_mass = [Va * aluminum_density,Vc * carbon_density]
    return r, r+aluminum_thickness, r+aluminum_thickness+carbon_thickness, aluminum_mass, carbon_mass

def cyl_tank_calc(volume,outermost_length,aluminum_thickness,carbon_thickness,cyl_guess,max_error):
    # lmax = l + 2ta + 2tc, Vcyl = pi*r^2*h, Vsph = (4/3)pi*r^3, V = pi*r^2(l-2r) + (4/3)pi*r^3, V = pi(lr^2 - 2r^3 + (4/3)r^3)
    cyl_length = outermost_length - (2*(aluminum_thickness+carbon_thickness))
    cyl_raph,f,f_ = cylinder_raphson_setup(cyl_guess,volume,cyl_length,max_error)
    r,iterations = cyl_raph(f,f_,cyl_guess)
    Va = (np.pi*(((cyl_length+(2*aluminum_thickness))*((r+aluminum_thickness)**2)) - ((2/3)*((r+aluminum_thickness)**3)))) - volume
    if Va < (10**-10):
        Va = 0
    Vc = (np.pi*(((cyl_length+(2*(aluminum_thickness+carbon_thickness)))*((r+aluminum_thickness+carbon_thickness)**2)) - ((2/3)*((r+aluminum_thickness+carbon_thickness)**3)))) - volume - Va
    if Vc < (10**-10):
        Vc = 0
    aluminum_mass,carbon_mass = [Va * aluminum_density,Vc * carbon_density]
    return r, r+aluminum_thickness, r+aluminum_thickness+carbon_thickness, aluminum_mass, carbon_mass

def prop_2sphO2_1torCH4(Delta_v,dry_mass,Isp,tor_outermost_radius,al_layer_thickness_tor,c_layer_thickness_tor,al_layer_thickness_sph,c_layer_thickness_sph,O2_CH4_mass_mixture_ratio,tor_guess,max_error,tank_mass_tor=0,tank_mass_sph=0):
    combined_mass = dry_mass + tank_mass_tor + tank_mass_sph
    propellant_mass = propellant_mass_calculation(Delta_v,combined_mass,Isp)
    Ox_volume,Fuel_volume = propellant_volume_calculation(propellant_mass,O2_CH4_mass_mixture_ratio)
    Ox_per_tank_vol,Fuel_per_tank_vol = [Ox_volume/2,Fuel_volume]
    rsph,rspha,rsphc,mspha,msphc = sph_tank_calc(Ox_per_tank_vol,al_layer_thickness_sph,c_layer_thickness_sph)
    rtor,rtora,rtorc,mtora,mtorc = tor_tank_calc(Fuel_per_tank_vol,tor_outermost_radius,al_layer_thickness_tor,c_layer_thickness_tor,tor_guess,max_error)
    Ox_comb_tank_mass = (2 * (mspha + msphc))
    Fuel_comb_tank_mass = (mtora + mtorc)
    # mass_addition_remainder_tor = (Fuel_comb_tank_mass - tank_mass_tor)
    # mass_addition_remainder_sph = (Ox_comb_tank_mass - tank_mass_sph)
    return rsph, rtor, Ox_comb_tank_mass, Fuel_comb_tank_mass, rtor #, mass_addition_remainder_tor, mass_addition_remainder_sph

def prop_2sphO2_2cylCH4(Delta_v,dry_mass,Isp,cyl_outermost_length,al_layer_thickness_cyl,c_layer_thickness_cyl,al_layer_thickness_sph,c_layer_thickness_sph,O2_CH4_mass_mixture_ratio,cyl_guess,max_error,tank_mass_cyl=0,tank_mass_sph=0):
    combined_mass = dry_mass + tank_mass_cyl + tank_mass_sph
    propellant_mass = propellant_mass_calculation(Delta_v,combined_mass,Isp)
    Ox_volume,Fuel_volume = propellant_volume_calculation(propellant_mass,O2_CH4_mass_mixture_ratio)
    Ox_per_tank_vol,Fuel_per_tank_vol = [Ox_volume/2,Fuel_volume/2]
    rsph,rspha,rsphc,mspha,msphc = sph_tank_calc(Ox_per_tank_vol,al_layer_thickness_sph,c_layer_thickness_sph)
    rcyl,rcyla,rcylc,mcyla,mcylc = cyl_tank_calc(Fuel_per_tank_vol,cyl_outermost_length,al_layer_thickness_cyl,c_layer_thickness_cyl,cyl_guess,max_error)
    Ox_comb_tank_mass = (2 * (mspha + msphc))
    Fuel_comb_tank_mass = (2 * (mcyla + mcylc))
    # mass_addition_remainder_cyl = (Fuel_comb_tank_mass - tank_mass_cyl)
    # mass_addition_remainder_sph = (Ox_comb_tank_mass - tank_mass_sph)
    return rsph, rcyl, Ox_comb_tank_mass, Fuel_comb_tank_mass, rcyl #, mass_addition_remainder_cyl, mass_addition_remainder_sph

def prop_2cylO2_2cylCH4(Delta_v,dry_mass,Isp,cyl_outermost_length_ox,cyl_outermost_length_fuel,al_layer_thickness_cyl_ox,c_layer_thickness_cyl_ox,al_layer_thickness_cyl_fuel,c_layer_thickness_cyl_fuel,O2_CH4_mass_mixture_ratio,cyl_guess_ox,cyl_guess_me,max_error,tank_mass_cyl_ox=0,tank_mass_cyl_fuel=0):
    combined_mass = dry_mass + tank_mass_cyl_ox + tank_mass_cyl_fuel
    propellant_mass = propellant_mass_calculation(Delta_v,combined_mass,Isp)
    Ox_volume,Fuel_volume = propellant_volume_calculation(propellant_mass,O2_CH4_mass_mixture_ratio)
    Ox_per_tank_vol,Fuel_per_tank_vol = [Ox_volume/2,Fuel_volume/2]
    rcyl_ox,rcyla_ox,rcylc_ox,mcyla_ox,mcylc_ox = cyl_tank_calc(Ox_per_tank_vol,cyl_outermost_length_ox,al_layer_thickness_cyl_ox,c_layer_thickness_cyl_ox,cyl_guess_ox,max_error)
    rcyl_me,rcyla_me,rcylc_me,mcyla_me,mcylc_me = cyl_tank_calc(Fuel_per_tank_vol,cyl_outermost_length_fuel,al_layer_thickness_cyl_fuel,c_layer_thickness_cyl_fuel,cyl_guess_me,max_error)
    Ox_comb_tank_mass = (2 * (mcyla_ox + mcylc_ox))
    Fuel_comb_tank_mass = (2 * (mcyla_me + mcylc_me))
    # mass_addition_remainder_cyl = (Fuel_comb_tank_mass - tank_mass_cyl_ox)
    # mass_addition_remainder_cyl = (Ox_comb_tank_mass - tank_mass_cyl_fuel)
    return rcyl_ox, rcyl_me, Ox_comb_tank_mass, Fuel_comb_tank_mass, rcyl_ox, rcyl_me #, mass_addition_remainder_cyl, mass_addition_remainder_sph

def prop_2sphO2_2sphCH4(Delta_v,dry_mass,Isp,cyl_outermost_length_ox,cyl_outermost_length_fuel,al_layer_thickness_cyl_ox,c_layer_thickness_cyl_ox,al_layer_thickness_cyl_fuel,c_layer_thickness_cyl_fuel,O2_CH4_mass_mixture_ratio,cyl_guess_ox,cyl_guess_me,max_error,tank_mass_cyl_ox=0,tank_mass_cyl_fuel=0):
    combined_mass = dry_mass + tank_mass_cyl_ox + tank_mass_cyl_fuel
    propellant_mass = propellant_mass_calculation(Delta_v,combined_mass,Isp)
    Ox_volume,Fuel_volume = propellant_volume_calculation(propellant_mass,O2_CH4_mass_mixture_ratio)
    Ox_per_tank_vol,Fuel_per_tank_vol = [Ox_volume/2,Fuel_volume/2]
    rcyl_ox,rcyla_ox,rcylc_ox,mcyla_ox,mcylc_ox = cyl_tank_calc(Ox_per_tank_vol,cyl_outermost_length_ox,al_layer_thickness_cyl_ox,c_layer_thickness_cyl_ox,cyl_guess_ox,max_error)
    rcyl_me,rcyla_me,rcylc_me,mcyla_me,mcylc_me = cyl_tank_calc(Fuel_per_tank_vol,cyl_outermost_length_fuel,al_layer_thickness_cyl_fuel,c_layer_thickness_cyl_fuel,cyl_guess_me,max_error)
    Ox_comb_tank_mass = (2 * (mcyla_ox + mcylc_ox))
    Fuel_comb_tank_mass = (2 * (mcyla_me + mcylc_me))
    # mass_addition_remainder_cyl = (Fuel_comb_tank_mass - tank_mass_cyl_ox)
    # mass_addition_remainder_cyl = (Ox_comb_tank_mass - tank_mass_cyl_fuel)
    return rcyl_ox, rcyl_me, Ox_comb_tank_mass, Fuel_comb_tank_mass, rcyl_ox, rcyl_me #, mass_addition_remainder_cyl, mass_addition_remainder_sph

#### Phase 2 Raptor 3 Engines
g_earth = 9.80665
aluminum_density = 2710 # kg/m^3
carbon_density = 1600 # kg/m^3
O2_CH4_mass_mixture_ratio = 3.6

dry_mass = 8922.656 # kg
Isp = 380
Delta_v = 5000
al_layer_thickness = 0
c_layer_thickness = 0.004
iterations = 6

initial_tank_mass_tor = 0 # equation must be iterated until the tanks account for there own mass
initial_tank_mass_sph = 0 # equation must be iterated until the tanks account for there own mass
tor_outermost_radius = 4 # The surface at the edge which would be carbon fiber. so outermost outermost edge
# combined_propellant_mass,propellant_system_mass = prop_mass_estimating(dry_mass,Isp,Delta_v,g_earth,O2_CH4_mass_mixture_ratio)
max_error = 10**-12
tor_guess = 0.6 # m
new_tank_mass_tor,new_tank_mass_sph = [initial_tank_mass_tor,initial_tank_mass_sph]
print("2 Oxygen Spheres & 1 Methane Toroidal:\n")
print("Diameter of the Toroidal: {} m".format(tor_outermost_radius*2))
for i in range(iterations):
    rsph, rtor, Ox_comb_tank_mass, Fuel_comb_tank_mass, tor_guess = prop_2sphO2_1torCH4(Delta_v,dry_mass,Isp,tor_outermost_radius,al_layer_thickness,c_layer_thickness,al_layer_thickness,c_layer_thickness,O2_CH4_mass_mixture_ratio,tor_guess,max_error,new_tank_mass_tor,new_tank_mass_sph)
    new_tank_mass_tor,new_tank_mass_sph = [Ox_comb_tank_mass, Fuel_comb_tank_mass]
    # print("Iteration = {}:".format(i+1))
propellant_mass = propellant_mass_calculation(Delta_v,dry_mass+new_tank_mass_tor+new_tank_mass_sph,Isp)
print("Sphere Radius = {} m\nToroidal Radius = {} m\nCombined Oxygen Tank Mass = {} kg\nCombined Fuel Tank Mass = {} kg\nTotal Tank Mass = {} kg\nTotal Propellant Mass = {} kg\n".format(rsph,rtor,Ox_comb_tank_mass,Fuel_comb_tank_mass,Ox_comb_tank_mass+Fuel_comb_tank_mass,propellant_mass))

initial_tank_mass_cyl = 0 # equation must be iterated until the tanks account for there own mass
initial_tank_mass_sph = 0 # equation must be iterated until the tanks account for there own mass
cyl_outermost_length = 6 # The surface at the edge which would be carbon fiber. so outermost outermost edge
max_error = 10**-12
cyl_guess = 0.7 # m
new_tank_mass_cyl,new_tank_mass_sph = [initial_tank_mass_cyl,initial_tank_mass_sph]
print("2 Oxygen Spheres & 2 Methane Cylinders:\n")
print("Length of the Cylinders: {} m".format(cyl_outermost_length))
for i in range(iterations):
    rsph, rcyl, Ox_comb_tank_mass, Fuel_comb_tank_mass, cyl_guess = prop_2sphO2_2cylCH4(Delta_v,dry_mass,Isp,cyl_outermost_length,al_layer_thickness,c_layer_thickness,al_layer_thickness,c_layer_thickness,O2_CH4_mass_mixture_ratio,cyl_guess,max_error,new_tank_mass_cyl,new_tank_mass_sph)
    new_tank_mass_cyl,new_tank_mass_sph = [Ox_comb_tank_mass, Fuel_comb_tank_mass]
    # print("Iteration = {}:".format(i+1))
propellant_mass = propellant_mass_calculation(Delta_v,dry_mass+new_tank_mass_cyl+new_tank_mass_sph,Isp)
print("Sphere Radius = {} m\nCylinder Radius = {} m\nCombined Oxygen Tank Mass = {} kg\nCombined Fuel Tank Mass = {} kg\nTotal Tank Mass = {} kg\nTotal Propellant Mass = {} kg\n".format(rsph,rcyl,Ox_comb_tank_mass,Fuel_comb_tank_mass,Ox_comb_tank_mass+Fuel_comb_tank_mass,propellant_mass))

initial_tank_mass_cyl = 0 # equation must be iterated until the tanks account for there own mass
cyl_outermost_length = 4 # The surface at the edge which would be carbon fiber. so outermost outermost edge
max_error = 10**-12
cyl_guess_ox,cyl_guess_me = [1,1] # m
new_tank_mass_cyl_ox,new_tank_mass_cyl_fuel = [initial_tank_mass_cyl,initial_tank_mass_cyl]
print("2 Oxygen Cylinders & 2 Methane Cylinders:\n")
print("Length of the Cylinders: {} m".format(cyl_outermost_length))
for i in range(iterations):
    rcyl_ox, rcyl_me, Ox_comb_tank_mass, Fuel_comb_tank_mass, cyl_guess_ox, cyl_guess_me = prop_2cylO2_2cylCH4(Delta_v,dry_mass,Isp,cyl_outermost_length,cyl_outermost_length,al_layer_thickness,c_layer_thickness,al_layer_thickness,c_layer_thickness,O2_CH4_mass_mixture_ratio,cyl_guess_ox,cyl_guess_me,max_error,new_tank_mass_cyl_ox,new_tank_mass_cyl_fuel)
    new_tank_mass_cyl_ox,new_tank_mass_cyl_me = [Ox_comb_tank_mass, Fuel_comb_tank_mass]
    # print("Iteration = {}:".format(i+1))
propellant_mass = propellant_mass_calculation(Delta_v,dry_mass+new_tank_mass_cyl_ox+new_tank_mass_cyl_me,Isp)
print("Cylinder Radius Oxygen = {} m\nCylinder Radius Fuel = {} m\nCombined Oxygen Tank Mass = {} kg\nCombined Fuel Tank Mass = {} kg\nTotal Tank Mass = {} kg\nTotal Propellant Mass = {} kg\n".format(rcyl_ox,rcyl_me,Ox_comb_tank_mass,Fuel_comb_tank_mass,Ox_comb_tank_mass+Fuel_comb_tank_mass,propellant_mass))

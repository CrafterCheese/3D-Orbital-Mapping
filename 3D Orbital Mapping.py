"""
------------------------------------------------------------
 File Name   : [3D Orbital Mapping]
 Author      : [CrafterCheese]
 Date Created: [2025-03-9] (YYYY-MM-DD)
 Description : [Maps an orbit in 3D space based on inputted parameters]

 Last Modified: [2025-07-21] by [CrafterCheese]
 Version       : [2.0]

 License       : [MIT]
------------------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Celestial body database
Celestial_Bodies = {
    "Mercury": {"mu": 22031.868551, "mass": 3.3011e23, "radius": 2439.7, "atmosphere": None, "a_sun": 57909227},
    "Venus": {"mu": 398600.435507, "mass": 4.8675e24, "radius": 6051.8, "atmosphere": 250, "a_sun": 108208000},
    "Earth": {"mu": 398600.4418, "mass": 5.972e24, "radius": 6371, "atmosphere": 100, "a_sun": 149600000},
    "Mars": {"mu": 42828.375816, "mass": 6.4171e23, "radius": 3389.5, "atmosphere": 50, "a_sun": 227939100},
    "Jupiter": {"mu": 126686511, "mass": 1.8982e27, "radius": 69911, "atmosphere": 500, "a_sun": 778340821},
    "Saturn": {"mu": 37931207.8, "mass": 5.6834e26, "radius": 58232, "atmosphere": 300, "a_sun": 1426666420},
    # Moon Database
    "Moon": {"mu": 4902.800066, "mass": 7.34767309e22, "radius": 1737.4, "atmosphere": None, "a_sun": 149600000},
    "Titan": {"mu": 8978.14, "mass": 1.3452e23, "radius": 2575, "atmosphere": 200, "a_sun": 1426666420},
    "Io": {"mu": 3660.0, "mass": 8.9319e22, "radius": 1821.6, "atmosphere": None, "a_sun": 778340821},
    "Europa": {"mu": 3200.0, "mass": 4.799e22, "radius": 1560.8, "atmosphere": None, "a_sun": 778340821},
    "Ganymede": {"mu": 9810.0, "mass": 1.4819e23, "radius": 2634.1, "atmosphere": None, "a_sun": 778340821},
    "Callisto": {"mu": 7030.0, "mass": 1.0759e23, "radius": 2410.3, "atmosphere": None, "a_sun": 778340821},
    "Rhea": {"mu": 4000.0, "mass": 2.312e23, "radius": 763.8, "atmosphere": None, "a_sun": 1426666420},
}

# Select Celestial Body
body_name = input("Enter the central body (e.g., Earth, Moon, Titan...): ")
if body_name in Celestial_Bodies:
    mu = Celestial_Bodies[body_name]["mu"]
    mass = Celestial_Bodies[body_name]["mass"]
    radius = Celestial_Bodies[body_name]["radius"]
    atmosphere = Celestial_Bodies[body_name]["atmosphere"]
    a_sun = Celestial_Bodies[body_name]["a_sun"]
    print(f"Using {body_name} with gravitational parameter mu = {mu} km^3/s^2")
else:
    print("Unknown body, Defaulting to Earth.")
    mu = Celestial_Bodies["Earth"]["mu"]
    mass = Celestial_Bodies["Earth"]["mass"]
    radius = Celestial_Bodies["Earth"]["radius"]
    atmosphere = Celestial_Bodies["Earth"]["atmosphere"]
    a_sun = Celestial_Bodies["Earth"]["a_sun"]
# Inputs
a = float(input("Enter semi-major axis (km): "))  # Semi-major axis in km
e = float(input("Enter eccentricity (0 to 1): "))  # Eccentricity
i = float(input("Enter Inclination (degrees): "))  # Inclination
raan = float(input("Enter right ascension of ascending node (degrees): "))
arg_periapsis = float(input("Enter argument of periapsis (degrees): "))
true_anomaly_object = float(input("Enter true anomaly of the object (degrees): "))
num_points = 20

# Convert Degrees to radians
i = np.radians(i)
raan = np.radians(raan)
arg_periapsis = np.radians(arg_periapsis)
true_anomaly_object = np.radians(true_anomaly_object)

theta = np.linspace(0, 2 * np.pi, 1000)  # True anomaly
true_anomaly_points = np.linspace(0, 2 * np.pi, num_points, endpoint=False)

# Compute orbit
r = a * (1 - e ** 2) / (1 + e * np.cos(theta))
x = r * np.cos(theta)
y = r * np.sin(theta)
z = np.zeros_like(x)

r_points = a * (1 - e ** 2) / (1 + e * np.cos(true_anomaly_points))
x_points = r_points * np.cos(true_anomaly_points)
y_points = r_points * np.sin(true_anomaly_points)
z_points = np.zeros_like(x_points)

r_object = a * (1 - e ** 2) / (1 + e * np.cos(true_anomaly_object))
x_object = r_object * np.cos(true_anomaly_object)
y_object = r_object * np.sin(true_anomaly_object)
z_object = 0
# Rotation Matrix
Rz_raan = np.array([[np.cos(raan), -np.sin(raan), 0],
                    [np.sin(raan), np.cos(raan), 0],
                    [0, 0, 1]])

Rz_i = np.array([[1, 0, 0],
                 [0, np.cos(i), -np.sin(i)],
                 [0, np.sin(i), np.cos(i)]])

Rz_arg_periapsis = np.array([[np.cos(arg_periapsis), -np.sin(arg_periapsis), 0],
                             [np.sin(arg_periapsis), np.cos(arg_periapsis), 0],
                             [0, 0, 1]])

# Rotation Matrix (RAAN -> Inclination -> Argument of Periapsis)
rotation_matrix = Rz_raan @ Rz_i @ Rz_arg_periapsis

# Transform Coordinates (orbit)
orbit_coords = np.vstack((x, y, z))
transformed_coords = rotation_matrix @ orbit_coords
x, y, z = transformed_coords

# Transform 20 Points
point_coords = np.vstack((x_points, y_points, z_points))
transformed_points = rotation_matrix @ point_coords
x_points, y_points, z_points = transformed_points

# Transform Object Position
object_coords = np.array([x_object, y_object, z_object])
x_object, y_object, z_object = rotation_matrix @ object_coords

# Output orbital points and object position
print("20 points on the orbital path evenly spaced")
for i in range(num_points):
    print(f"point {i + 1}: X = {x_points[i]:.2f} km, Y = {y_points[i]:.2f} km, Z = {z_points[i]:.2f} km")

print(f"Object Position: X = {x_object:.2f} km, Y = {y_object:.2f} km, Z = {z_object:.2f} km")

# Plotting the 3D Orbital Path
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, label='Orbital Path')

# Scatter central body at origin
ax.scatter([0], [0], [0], color="red", label="Central Body")

# Scatter the 20 points on orbital path
ax.scatter(x_points, y_points, z_points, color="blue", label="20 points")

# Scatter object position
ax.scatter(x_object, y_object, z_object, color="green", label="Object Position", zorder=3)

# Points
for i in range(num_points):
    ax.text(x_points[i], y_points[i], z_points[i], str(i + 1), fontsize=8)
ax.text(x_object, y_object, z_object, "Object", fontsize=10, color='green')

# Plot the planet surface as a sphere
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)

# Use the radius to create a sphere (for planet's surface)
planet_x = radius * np.outer(np.cos(u), np.sin(v))
planet_y = radius * np.outer(np.sin(u), np.sin(v))
planet_z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(planet_x, planet_y, planet_z, color='b')

# Check atmosphere and position
distance_from_center = np.sqrt(x_object ** 2 + y_object ** 2 + z_object ** 2)
if atmosphere is not None and distance_from_center <= (radius + atmosphere):
    print(f"Object is inside the atmosphere of {body_name}.")
else:
    print(f"Object is outside the atmosphere of {body_name}")

# Atmosphere plotting
if atmosphere is not None:
    atmosphere_radius = radius + atmosphere

    atmosphere_x = atmosphere_radius * np.outer(np.cos(u), np.sin(v))
    atmosphere_y = atmosphere_radius * np.outer(np.sin(u), np.sin(v))
    atmosphere_z = atmosphere_radius * np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(atmosphere_x, atmosphere_y, atmosphere_z, color='r', alpha=0.2)

# Set plot limits to focus on the orbit and the planet
ax.set_xlim([-1.5 * a, 1.5 * a])  # X-axis range centered around the orbit
ax.set_ylim([-1.5 * a, 1.5 * a])  # Y-axis range centered around the orbit
ax.set_zlim([-1.5 * a, 1.5 * a])  # Z-axis range centered around the orbit

# All axes are scaled equally for a sphere
ax.set_box_aspect([1, 1, 1])  # Makes aspect ratio is equal for X, Y, Z

# labels, titles, and legend for the plot
ax.set_xlabel("X Position (km)")
ax.set_ylabel("Y Position (km)")
ax.set_zlabel("Z Position (km)")
ax.set_title(f"3D Elliptical Orbit around {body_name}")
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()

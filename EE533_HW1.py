"""
DC Power Flow Solution for 5-Bus System
Author: Jeff Dinsmore style format requested
Date: 04/07/2026

Description:
------------
This script solves the DC power flow equations for the 5-bus network shown
in the homework problem.

Assumptions confirmed:
- Base MVA = 100
- Bus A is the slack bus
- Slack angle theta_A = 0 radians
- Branch reactances are taken from the imaginary parts of the line impedances
- Branch flow sign convention:
    P_ij > 0  means power flows from bus i to bus j
    P_ij < 0  means actual flow is from bus j to bus i

Outputs:
--------
a) The reduced P vector used in the calculations
b) Final bus voltage angles in degrees
c) Branch flows in MW
"""

import numpy as np


# ============================================================
# 1. SYSTEM DATA
# ============================================================

# Base power
base_mva = 100.0

# Bus order used throughout this script:
# A = 0, B = 1, C = 2, D = 3, E = 4
bus_names = ["A", "B", "C", "D", "E"]

# Slack bus
slack_bus = 0  # Bus A

# ------------------------------------------------------------
# Branch reactances X_ij = imag(Z_ij)
# Stored as: (from_bus, to_bus): X
# ------------------------------------------------------------
branches = {
    ("A", "B"): 0.10,
    ("A", "D"): 0.60,
    ("A", "E"): 0.15,
    ("B", "C"): 0.50,
    ("C", "D"): 0.10,
    ("E", "D"): 0.30,
}

# ------------------------------------------------------------
# Net real power injections in MW
# P_i = P_Gi - P_Li
#
# Bus A = slack bus, so its P is not specified in the reduced solve
# Bus B = -80 MW
# Bus C =   0 MW
# Bus D =  50 - 220 = -170 MW
# Bus E = -125 MW
# ------------------------------------------------------------
P_mw_full = np.array([
    0.0,     # Bus A -> placeholder only, not used in reduced solve
    -80.0,   # Bus B
    0.0,     # Bus C
    -170.0,  # Bus D
    -125.0   # Bus E
], dtype=float)

# Convert MW to per unit
P_pu_full = P_mw_full / base_mva


# ============================================================
# 2. BUILD FULL B_BUS MATRIX
# ============================================================

n_bus = len(bus_names)
bus_index = {name: idx for idx, name in enumerate(bus_names)}

# Initialize B_bus
B_bus = np.zeros((n_bus, n_bus), dtype=float)

# For each line:
#   off-diagonal = -1/X
#   diagonal     = sum(1/X) of connected branches
for (from_bus, to_bus), x in branches.items():
    i = bus_index[from_bus]
    j = bus_index[to_bus]

    bij = 1.0 / x

    B_bus[i, i] += bij
    B_bus[j, j] += bij
    B_bus[i, j] -= bij
    B_bus[j, i] -= bij


# ============================================================
# 3. REDUCE MATRIX BY REMOVING SLACK BUS
# ============================================================

# Create list of non-slack bus indices
non_slack = [i for i in range(n_bus) if i != slack_bus]

# Reduced B' matrix
B_reduced = B_bus[np.ix_(non_slack, non_slack)]

# Reduced P vector in per unit
P_reduced_pu = P_pu_full[non_slack]


# ============================================================
# 4. SOLVE FOR UNKNOWN BUS ANGLES
# ============================================================

# Solve:
#     P_reduced = B_reduced * theta_reduced
theta_reduced_rad = np.linalg.solve(B_reduced, P_reduced_pu)

# Build full theta vector, including slack angle = 0
theta_full_rad = np.zeros(n_bus, dtype=float)
theta_full_rad[non_slack] = theta_reduced_rad

# Convert to degrees
theta_full_deg = np.degrees(theta_full_rad)


# ============================================================
# 5. COMPUTE SLACK BUS REAL POWER INJECTION
# ============================================================

# Using full B matrix:
#     P = B_bus * theta
P_calc_pu = B_bus @ theta_full_rad
P_calc_mw = P_calc_pu * base_mva

# Slack bus injection in MW
P_slack_mw = P_calc_mw[slack_bus]


# ============================================================
# 6. COMPUTE BRANCH FLOWS
# ============================================================

# DC branch flow:
#     P_ij(pu) = (theta_i - theta_j) / X_ij
#     P_ij(MW) = P_ij(pu) * base_mva
branch_flows_mw = {}

for (from_bus, to_bus), x in branches.items():
    i = bus_index[from_bus]
    j = bus_index[to_bus]

    p_ij_pu = (theta_full_rad[i] - theta_full_rad[j]) / x
    p_ij_mw = p_ij_pu * base_mva

    branch_flows_mw[(from_bus, to_bus)] = p_ij_mw


# ============================================================
# 7. PRINT RESULTS
# ============================================================

print("=" * 60)
print("DC POWER FLOW RESULTS")
print("=" * 60)

# ------------------------------------------------------------
# a) Reduced P vector used in calculations
# ------------------------------------------------------------
print("\na) Reduced P vector used in the calculations")
print("   (Non-slack buses only: B, C, D, E)")
print("   In MW:")
print(f"   P_reduced_MW = {P_mw_full[non_slack]}")

print("\n   In per unit on 100 MVA base:")
print(f"   P_reduced_pu = {P_reduced_pu}")

# Optional: show full net injection vector after solve
print("\n   Full bus injection vector after solving (MW):")
for i, name in enumerate(bus_names):
    print(f"   Bus {name}: {P_calc_mw[i]: .4f} MW")

# ------------------------------------------------------------
# b) Final bus angles
# ------------------------------------------------------------
print("\nb) Final bus voltage angles")
for i, name in enumerate(bus_names):
    print(f"   Theta_{name} = {theta_full_deg[i]: .6f} degrees")

# ------------------------------------------------------------
# c) Branch flows
# ------------------------------------------------------------
print("\nc) Branch flows (positive = listed direction, negative = reverse)")
for (from_bus, to_bus), flow_mw in branch_flows_mw.items():
    print(f"   P_{from_bus}{to_bus} = {flow_mw: .6f} MW")

print("\n" + "=" * 60)
print("MATRICES")
print("=" * 60)

print("\nFull B_bus matrix:")
print(B_bus)

print("\nReduced B' matrix (slack bus removed):")
print(B_reduced)
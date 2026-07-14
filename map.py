import osmnx as ox
import matplotlib.pyplot as plt

# Download Karachi road network
place_name = "Karachi, Sindh, Pakistan"
G = ox.graph_from_place(place_name, network_type="drive")

# Create a large figure
fig, ax = plt.subplots(figsize=(16, 16))

# Plot on the axes
ox.plot_graph(
    G,
    ax=ax,
    node_size=0,
    edge_color="black",
    edge_linewidth=0.4,
    bgcolor="white",
    show=False,
    close=False
)

# Remove axes and all padding
ax.set_axis_off()
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

# Maximize the window (Windows)
manager = plt.get_current_fig_manager()
try:
    manager.window.state("zoomed")
except:
    try:
        manager.window.showMaximized()
    except:
        pass

plt.show()
from fink_client.visualisation import extract_field
from fink_client.visualisation import show_stamps

import matplotlib
import matplotlib.pyplot as plt

import numpy as np

# Bands
filter_color = {1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c'}
filter_name = {1: 'g band', 2: 'r band', 3: 'i band'}

def plot_alert_data(alert: dict) -> None:
    """ Plot alert data (stamps and lightcurve)

    Parameters
    ----------
    alert: dict
        Dictionary containing alert data.
    """
    fig = plt.figure(num=0, figsize=(12, 4))
    show_stamps(alert, fig)
    plt.show()

    # extract current and historical data as one vector
    mag = extract_field(alert, 'magpsf')
    error = extract_field(alert, 'sigmapsf')
    upper = extract_field(alert, "diffmaglim")

    # filter bands
    fid = extract_field(alert, "fid")

    # Rescale dates to end at 0
    jd = extract_field(alert, "jd")
    dates = np.array([i - jd[0] for i in jd])

    # Title of the plot (alert ID)
    title = alert["objectId"]

    # loop over filters
    fig = plt.figure(num=1, figsize=(12, 4))

    # Loop over each filter
    for filt in filter_color.keys():
        mask = np.where(fid == filt)[0]

        # Skip if no data
        if len(mask) == 0:
            continue

        # y data
        maskNotNone = mag[mask] != None
        plt.errorbar(
            dates[mask][maskNotNone], mag[mask][maskNotNone],
            yerr=error[mask][maskNotNone],
            color=filter_color[filt], marker='o',
            ls='', label=filter_name[filt], mew=4)
        # Upper limits
        plt.plot(
            dates[mask][~maskNotNone], upper[mask][~maskNotNone],
            color=filter_color[filt], marker='v', ls='', mew=4, alpha=0.5)
        plt.title(title)
    plt.legend()
    plt.gca().invert_yaxis()
    plt.xlabel('Days to candidate')
    plt.ylabel('Difference magnitude')
    plt.show()
    
def spider_plot(labels, values, tot, ticks, legend='', ax=None) -> None:
    """ Make a spider plot.
    
    Parameters
    ----------
    labels: list
        List of labels (theta)
    values: list
        List of values (r)
    tot: int
        Total number of entries to normalise `values`
    legend: str
        Legend for the plot
    ax: plt.ax
        Figure ax (for multiple plots)
    """
    # number of variable
    N = len(labels)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    # If you want the first axis to be on top:
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], labels)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks(ticks, ['{}%'.format(i) for i in ticks], color="black", size=15)


    # ------- PART 2: Add plots
    values = values.tolist()
    values += values[:1]
    values = [i/tot*100 for i in values]
    g = ax.plot(angles, values, linewidth=1, linestyle='solid', label=legend)
    ax.fill(angles, values, g[0].get_color(), alpha=0.2)
    ax.set_ylim(0,ticks[-1]+1)
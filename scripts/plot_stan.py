import matplotlib.pyplot as plt
import json
import sys

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_trace_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    with open(file_path, 'r') as file:
        data = []
        for line in file:
            json_line = json.loads(line)

            if json_line['rollout_num'] != 0:
                break
            
            data.append(json_line)

    # Extracting the x and y coordinates for both the lead and wingman
    lead_x = [entry['info']['lead']['x'] for entry in data]
    lead_y = [entry['info']['lead']['y'] for entry in data]
    wingman_x = [entry['info']['wingman']['x'] for entry in data]
    wingman_y = [entry['info']['wingman']['y'] for entry in data]
    wingman_speed = [entry['info']['wingman']['v'] for entry in data]
    wingman_throttle = [entry['info']['wingman']['controller']['control'][1] for entry in data]
    wingman_untrimmed_throttle = [entry['info']['wingman']['controller']['untrimmed_control'][1] for entry in data]

    # Setting up subplots
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], width_ratios=[2, 1])

    ax1 = fig.add_subplot(gs[:, 0])  # Trajectory plot
    ax2 = fig.add_subplot(gs[0, 1])  # Speed plot
    ax3 = fig.add_subplot(gs[1, 1])  # Throttle plot
    
    # Plotting the trajectories
    ax1.plot(lead_x, lead_y, label='Lead', marker='o')
    ax1.plot(wingman_x, wingman_y, label='Follower', marker='o')
    ax1.scatter([lead_x[0], lead_x[-1]], [lead_y[0], lead_y[-1]], color='red')  # Start and End points for Lead
    ax1.scatter([wingman_x[0], wingman_x[-1]], [wingman_y[0], wingman_y[-1]], color='blue')  # Start and End points for Wingman
    ax1.text(lead_x[0], lead_y[0], 'Start')
    ax1.text(lead_x[-1], lead_y[-1], 'End')
    ax1.text(wingman_x[0], wingman_y[0], 'Start')
    ax1.text(wingman_x[-1], wingman_y[-1], 'End')
    ax1.set_title('Trajectory of Aircraft')
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.legend()
    ax1.grid(True)

    # Plotting the speed of the follower
    ax2.plot(wingman_speed, 'g-', label='Follower Speed', marker='o')
    ax2.set_title('Speed of the Follower')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Speed')
    ax2.grid(True)

    # Plotting the throttle command
    ax3.plot(wingman_throttle, 'b-', label='Throttle')
    ax3.plot(wingman_untrimmed_throttle, 'b--', label='Untrimmed')
    ax3.set_title('Throttle Command to Follower')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Throttle Command')
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    
    plt.savefig('eval_plot.png')
    plt.show()

if __name__ == "__main__":
    main()

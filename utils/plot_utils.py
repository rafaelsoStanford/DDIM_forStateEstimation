import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap
from PIL import Image
import io
import torch
from matplotlib.animation import FuncAnimation


def plt2tsb(figure, writer, fig_name, niter):
    # Save the plot to a BytesIO object
    buf = io.BytesIO()
    figure.savefig(buf, format='png')
    buf.seek(0)

    # Open the image and convert to RGB, then to Tensor
    image = Image.open(buf).convert('RGB')
    image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1)

    # Add the image to TensorBoard
    writer.add_image(fig_name, image_tensor, niter)
    buf.close()


def plt_toTensorboard(self,
    position_observation,   
    positions_groundtruth,
    positions_predicted,
    actions_predicted,
    actions_groundtruth,
    actions_observation):
    # ---------------- Plotting ----------------
    writer = self.logger.experiment
    niter  = self.global_step
    plt.switch_backend('agg')
    # ---------------- 2D Position Plot ----------------
    fig = plt.figure()
    fig.clf()
    # Create a colormap for fading colors based on the number of timesteps
    cmap = get_cmap('viridis', self.pred_horizon + self.inpaint_horizon)
    # Create an array of indices from 0 to timesteps-1
    indices = np.arange(self.pred_horizon + self.inpaint_horizon)
    # Normalize the indices to the range [0, 1]
    normalized_indices = indices / (self.pred_horizon + self.inpaint_horizon - 1)
    # Create a color array using the colormap and normalized indices
    colors = cmap(normalized_indices)

    plt.plot(position_observation[:, 0], position_observation[:,1],'b.')
    plt.plot(positions_groundtruth[self.inpaint_horizon:,0], positions_groundtruth[self.inpaint_horizon:,1],'g.')
    plt.scatter(positions_predicted[:,0],positions_predicted[:,1],color=colors, s = 20)

    plt.grid()
    plt.axis('equal')

    # Plot to tensorboard
    plt2tsb(fig, writer, 'Predicted_path ' + self.date , niter)
    
    # ---------------- Action space Plotting ----------------
    # Visualize the action data
    inpaint_start = 0
    inpaint_end = self.inpaint_horizon

    fig2, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.plot(actions_predicted[:,0])
    ax1.plot(actions_groundtruth[:,0])
    ax1.scatter(np.arange(actions_predicted.shape[0]), actions_predicted[:,0] , c='r', s=10)
    ax1.axvspan(inpaint_start, inpaint_end, alpha=0.2, color='red')
    ax1.axvspan(inpaint_end, actions_predicted.shape[0], alpha=0.2, color='green')

    ax2.plot(actions_predicted[:,1])
    ax2.plot(actions_groundtruth[:,1])
    ax2.scatter(np.arange(actions_predicted.shape[0]), actions_predicted[:,1] , c='r', s=10)
    ax2.axvspan(inpaint_start, inpaint_end, alpha=0.2, color='red')
    ax2.axvspan(inpaint_end, actions_predicted.shape[0], alpha=0.2, color='green')

    ax3.plot(actions_predicted[:,2])
    ax3.plot(actions_groundtruth[:,2])
    ax3.scatter(np.arange(actions_predicted.shape[0]), actions_predicted[:,2] , c='r', s=10)
    ax3.axvspan(inpaint_start, inpaint_end, alpha=0.2, color='red')
    ax3.axvspan(inpaint_end, actions_predicted.shape[0], alpha=0.2, color='green')
    plt2tsb(fig2, writer, 'Action comparisons' + self.date , niter)

    plt.close('all')


def plt_toVideo(self, 
                sampling_history,
                position_observation,   
                positions_groundtruth, 
                actions_groundtruth ,
                actions_observation):
    # ---------------- Plotting ----------------
    # 
    sampling_positions = np.array(sampling_history)[:, :, :2]  # (1000, 45 , 2)
    sampling_actions = np.array(sampling_history)[:, :, 2:]  # (1000, 45 , 3)

    def plot_positions():
        fig, ax = plt.subplots()

        cmap = plt.get_cmap('viridis', self.pred_horizon + self.inpaint_horizon)
        indices = np.arange(self.pred_horizon + self.inpaint_horizon)

        def animate(frame):
            fig.clf()
            normalized_indices = indices / (self.pred_horizon + self.inpaint_horizon - 1)
            colors = cmap(normalized_indices)

            plt.plot(position_observation[:, 0], position_observation[:, 1], 'b.')
            plt.plot(positions_groundtruth[self.inpaint_horizon:, 0], positions_groundtruth[self.inpaint_horizon:, 1], 'g.')
            plt.scatter(sampling_positions[frame, :, 0], sampling_positions[frame, :, 1], color=colors, s=20)

            plt.grid()
            plt.axis('equal')
            plt.xlim(-1.5, 1.5)
            plt.ylim(-1.5, 1.5)

        fig.animation = FuncAnimation(fig, animate, frames=self.denoising_steps, interval=50, repeat=False)
        fig.animation.save('./animations/' + self.date + 'animation_positions.mp4')
        print("Animation saved")
        plt.close('all')

    def plot_actions():
        fig2, ax1 = plt.subplots()

        def animate_actions(frame):
            fig2.clf()

            plt.plot(actions_groundtruth[:, 0])
            # ax2.plot(actions_groundtruth[ :, 1])
            # ax3.plot(actions_groundtruth[ :, 2])
            inpaint_start = 0
            inpaint_end = self.inpaint_horizon
            plt.axvspan(inpaint_start, inpaint_end, alpha=0.2, color='red')
            plt.axvspan(inpaint_end, sampling_actions.shape[1], alpha=0.2, color='green')
            #ax1.plot(sampling_actions[frame, :, 0])
            plt.scatter(np.arange(sampling_actions.shape[1]), sampling_actions[frame,:,0] , c='r', s=10)

            # ax3.plot(sampling_actions[frame, :, 2])
            plt.grid()
            plt.ylim(-1.5, 1.5)

        fig2.animation = FuncAnimation(fig2, animate_actions, frames=self.denoising_steps, interval=50, repeat=False)
        fig2.animation.save('./animations/' + self.date + 'animation_actions.gif', writer='pillow')
        print("Animation saved")
        plt.close('all')

    plot_actions()
    plot_positions()
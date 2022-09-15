from src.env.mdps.MDPGridWorld import MDPGridWorld


def tst_MDPGridWorld():
    y_max = 3
    x_max = 4
    p_noise = 0.1
    mdp_grid_world = MDPGridWorld(y_max=y_max, x_max=x_max, p_noise=p_noise)
    print(mdp_grid_world.debug_str())
    print(mdp_grid_world.P)


if __name__ == "__main__":
    tst_MDPGridWorld()
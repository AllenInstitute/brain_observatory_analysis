import seaborn as sns


session_order = {"TRAINING_0_gratings_autorewards_15min": 0,
                 "TRAINING_1_gratings": 1,
                 "TRAINING_2_gratings_flashed": 2,
                 "TRAINING_3_images_A_10uL_reward": 3,
                 "TRAINING_4_images_A_training": 4,
                 "TRAINING_5_images_A_epilogue": 5,
                 "TRAINING_5_images_A_handoff_ready": 6,
                 "OPHYS_1_images_A": 7,
                 "OPHYS_4_images_B": 8,
                 "OPHYS_6_images_B": 9,
                 }

########################################################################
# Short Session variables
########################################################################

# quick way to generate colors
# colors = sns.color_palette("tab10", len(short_session_order))
# short_session_order_colors = dict(zip(short_session_order.keys(), colors))

short_session_order_colors = {"Auto Rewards": sns.color_palette('Purples_r', 6)[:5][::2][1],
                              "Static Gratings": sns.color_palette('Greens_r', 6)[:5][::2][1],
                              "Flashed Gratings": sns.color_palette('Greens_r', 6)[:5][::2][2],
                              "Flashed Images": sns.color_palette('Reds_r', 6)[:5][::2][1],
                              "Familiar Images + omissions": sns.color_palette('Reds_r', 6)[:5][::2][2],
                              "Novel Images + omissions": sns.color_palette('Blues_r', 6)[:5][::2][1],
                              "Novel Images EXTINCTION": sns.color_palette('Blues_r', 6)[:5][::2][2]
                              }

short_session_order_colors = {"Auto Rewards": sns.color_palette('Purples_r', 6)[:5][::2][1],
                              "Static Gratings": sns.color_palette('Greens_r', 6)[:5][::2][1],
                              "Flashed Gratings": sns.color_palette('Greens_r', 6)[:5][::2][2],
                              "Flashed Images": sns.color_palette('Reds_r', 6)[:5][::2][1],
                              "Familiar Images + omissions": sns.color_palette('Reds_r', 6)[:5][::2][2],
                              "Novel Images + omissions": sns.color_palette('Blues_r', 6)[:5][::2][1],
                              "Novel Images EXTINCTION": sns.color_palette('Blues_r', 6)[:5][::2][2]
                              }

short_session_type_map = {"TRAINING_0_gratings_autorewards_15min": "Auto Rewards",
                          "TRAINING_1_gratings": "Static Gratings",
                          "TRAINING_2_gratings_flashed": "Flashed Gratings",
                          "TRAINING_3_images_A_10uL_reward": "Flashed Images",
                          "TRAINING_4_images_A_training": "Flashed Images",
                          "TRAINING_5_images_A_epilogue": "Flashed Images",
                          "TRAINING_5_images_A_handoff_ready": "Flashed Images",
                          "OPHYS_1_images_A": "Familiar Images + omissions",
                          "OPHYS_4_images_B": "Novel Images + omissions",
                          "OPHYS_6_images_B": "Novel Images EXTINCTION",
                          }

short_session_order_breaks = {"Auto Rewards": 0,
                              "Static Gratings": 1,
                              "Flashed Gratings": 2,
                              "Flashed Images": 3,
                              "Familiar Images \n + omissions": 4,
                              "Novel Images \n + omissions": 5,
                              "Novel Images \n EXTINCTION": 6
                              }

short_session_type_order = {"Auto Rewards": 0,
                            "Static Gratings": 1,
                            "Flashed Gratings": 2,
                            "Flashed Images": 3,
                            "Familiar Images + omissions": 4,
                            "Novel Images + omissions": 5,
                            "Novel Images EXTINCTION": 6
                            }

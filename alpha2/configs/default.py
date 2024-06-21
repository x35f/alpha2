default_args = {
    "num_mcts_actors": 3,
    "task_spec":{ 
        "max_expression_size": 10, # max length of alpha expression
    },
    "buffer":{
        "max_buffer_size": 1000,
    },
    "shared_storage":{
        'max_num_networks': 3,
    },

    "trainer":{
        "snapshot_interval": 150,
        "update_target_network_interval": 50,
        "num_network_training_steps": 1000000,
        "start_training_buffer_size": 10,
        "batch_size": 128,
        "num_td_steps": 3,
        "lr":{
            "init_value": 2e-4,
            "momentum": 0.9
        },
    },
    "mcts":{
        "visit_softmax_temperature_fn": "lambda steps: ( 1.0 if steps < 500e3 else 0.5 if steps < 750e3 else 0.25)",
        "num_simulations": 200000,
        "discount_factor": 1.0,
        # root prior
        "root_dirichlet_alpha": 0.1,
        "root_exploration_fraction": 0.25,

        # UCB related
        "pb_c_base": 20000,
        "pb_c_init": 1.0,

        #node
        "topk": 5

    },
    "network":{
        "pretrain_net_path": "",
        "embedding_dim": 128,
        "representation":{
            "num_resnet_blocks": 8,
            "use_program_encoding": True,
            "use_location_encoding": False,
            "use_permutation_embedding": False
        },
        "attention":{
            "key_size": 256,
            "num_heads": 8,
            "attention_dropout": False,
            "position_encoding": "absolute",
            "initializer": "Orthogonal",
            "num_layers": 4
        },
        "resnet":{
            "attention_dropout": False,
            "use_projection": True,
            "num_layers": 4
        },
        "prediction":{ # discretized value net
            "value_min": -0.2,
            "value_max": 0.2,
            "num_bins": 30
        }
    }
}
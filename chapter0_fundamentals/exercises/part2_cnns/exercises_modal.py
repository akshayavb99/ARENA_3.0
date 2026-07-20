import modal

from plotly_utils import line
from part2_cnns.exercises import (
    Conv2d,
    SimpleMLPTrainingArgs,
    app,
    test_mnist,
    training_loop,
    ReLU,
    Linear,
    SimpleMLP,
    train,
)

from part2_cnns import tests


if __name__ == "__main__":
    # local tests
    tests.test_relu(ReLU)
    tests.test_linear_parameters(Linear, bias=False)
    tests.test_linear_parameters(Linear, bias=True)
    tests.test_linear_forward(Linear, bias=False)
    tests.test_linear_forward(Linear, bias=True)
    tests.test_mlp_module(SimpleMLP)
    tests.test_mlp_forward(SimpleMLP)
    test_mnist.local() # Run with test_mnist.remote() to run on Modal
    training_loop.local()
    
    args = SimpleMLPTrainingArgs()
    loss_list, accuracy_list, model = train(args)
    
    tests.test_conv2d_module(Conv2d)
    m = Conv2d(in_channels=24, out_channels=12, kernel_size=3, stride=2, padding=1)
    print(f"Manually verify that this is an informative repr: {m}")

    
            
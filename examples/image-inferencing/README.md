# Image Inferencing Example

This README provides an overview of the image inferencing example, which demonstrates how to utilize the image processing capabilities of the application. The example showcases how to send an image URL to a model and receive a descriptive response about the image.

This feature only works with OpenAI models, including gpt-4o, gpt-4o-mini or gpt-4-turbo. To extend this to other models, please raise an issue on GitHub or contact us.

## Overview

The image inferencing example is designed to illustrate how to interact with the model using an image as input. The model processes the image and generates a textual description based on its content. This can be useful for applications such as image recognition, content generation, and more.

## Example Flow

The example flow consists of a series of nodes that define how the input image is processed and how the model generates a response. Below is a breakdown of the key components:

1. **Prompt Node**: This node initiates the process by sending a prompt to the model. The prompt includes the image URL and asks the model to describe the image.

2. **LLM Settings**: The settings for the model are defined, including parameters such as `temperature`, `response_format`, and `system_msg`. These settings control how the model generates its responses.

3. **Response Handling**: The responses from the model are captured and can be displayed or processed further. The example includes handling for both successful responses and potential errors.

## Key Features

- **Dynamic Image Input**: The example allows for dynamic input of image URLs, enabling users to test various images and see how the model responds.
- **Customizable Model Settings**: Users can adjust the model settings to fine-tune the output, such as changing the `temperature` for more creative or conservative responses.
- **Comprehensive Response Structure**: The example captures detailed responses from the model, including the raw response data, which can be useful for debugging or further analysis.


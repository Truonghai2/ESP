import argparse
import os


def convert_tflite_to_header(tflite_path, output_path):
    # Read the TFLite model file
    with open(tflite_path, 'rb') as f:
        model_data = f.read()
    
    # Convert to hex array
    hex_array = ', '.join([f'0x{b:02x}' for b in model_data])
    
    # Create header file content
    header_content = f"""#ifndef MODEL_DATA_H
#define MODEL_DATA_H

// Model data
const unsigned char fire_detection_model_tflite[] = {{
    {hex_array}
}};

const unsigned int fire_detection_model_tflite_len = sizeof(fire_detection_model_tflite);

#endif // MODEL_DATA_H
"""
    
    # Write to output file
    with open(output_path, 'w') as f:
        f.write(header_content)
    
    print(f"Model converted successfully! Output saved to {output_path}")
    print(f"Model size: {len(model_data)} bytes")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert TFLite model to C++ header')
    parser.add_argument('input', help='Input TFLite model file path')
    parser.add_argument('output', help='Output header file path')
    args = parser.parse_args()
    
    convert_tflite_to_header(args.input, args.output) 
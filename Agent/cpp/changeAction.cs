#include <stdio.h>
#include <math.h>
#include <stdint.h>

uint64_t encode_float_array(const float* array, int action_size)
{
    uint64_t result = 0;
    uint64_t sign_bits = 0;
    uint64_t value_bits = 0;

    for (int i = 0; i < action_size; ++i)
    {
        // Extract sign bit and make the number positive
        int sign_bit = array[i] < 0 ? 1 : 0;
        float positive_value = fabs(array[i]);

        // Shift the number left by 12 bits and convert to integer
        uint64_t shifted_value = (uint64_t)(positive_value * pow(2, 12));

        // Append the sign bit to sign_bits
        sign_bits = (sign_bits << 1) | sign_bit;

        // Append the shifted value to value_bits
        value_bits = (value_bits << 12) | shifted_value;
    }

    // Combine the sign bits and value bits
    // Sign bits are placed in the front
    result = (sign_bits << (action_size * 12)) | value_bits;

    return result;
}

int main()
{
    int action_size = 4;
    float action_array[] = { 1.5f, -2.3f, 3.6f, -4.1f };

    uint64_t encoded_value = encode_float_array(action_array, action_size);
    printf("Encoded Value: %llu\n", encoded_value);

    return 0;
}

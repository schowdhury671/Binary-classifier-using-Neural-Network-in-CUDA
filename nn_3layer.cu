#include <bits/stdc++.h>
#include <cuda.h>
#include "parallel_utility.cu"

#define DATA_DIM 3072
#define TRAIN_SAMPLE 500
#define TEST_SAMPLE 121

#define OUTPUT_LAYER_NODES 1
#define HIDDEN_LAYER_NODES 5

#define EPOCH 20
#define LEARNING_RATE_IP_HIDDEN 1
#define LEARNING_RATE_HIDDEN_OP 1

using namespace std;

void print2Dmat(double **arr, int m, int n) {
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            // cout << arr[i][j] << " ";
            printf("%0.9f ", arr[i][j]);
        }
        // Newline for new row
        cout << endl;
    }
    cout << endl;
}

void print1Dmat(double *arr, int m) {
    for (int i = 0; i < m; i++)
        cout << arr[i] << " ";

    cout << endl;
}

/*
void init_2D_mat(double **(&arr), int row, int col) {
    arr = (double **)malloc(row * sizeof(double *));
    for (int i = 0; i < row; i++)
        arr[i] = (double *)malloc(col * sizeof(double));
}
*/

void init_1D_mat(double *(&arr), int n) {
    arr = (double *)malloc(n * sizeof(double));
}

void load_data(double **(&data), double *labels, int row, int col, const char *filename){
    // Read Training Data
    FILE *fp = fopen(filename, "r");
    float ch;
    fscanf(fp, "%f", &ch);
    int i = 0, j = 0, ct = 0;
    double **dataset;
    init_2D_mat(dataset, row, col);
    
    while (ct != (row * col))
    {
        dataset[i][j++] = ch;
        if (j == col)
        {
            i++;
            j = 0;
        }
        fscanf(fp, "%f", &ch);
        ct++;
    }
    fclose(fp);

    for(i = 0; i < row; ++i) {
        for (j = 0; j < col - 1; ++j)
            data[i][j] = dataset[i][j];
        labels[i] = dataset[i][j];
    }
    
}

void init_weights(double **(&w), int row, int col) {
    for (int i = 0; i < row; i++)
        for (int j = 0; j < col; j++)
            w[i][j] = ((double)rand() / (double)RAND_MAX);
}

void init_biases(double *(&b), int row) {
    for (int i = 0; i < row; i++)
        b[i] = ((double)rand() / (double)RAND_MAX);
}

/********************************************************************/

double **mat_multiply(double **(&a), double **(&b), int r1, int c1, int r2, int c2) {
    double **c;
    init_2D_mat(c, r1, c2);
    // Multiplying matrix a and b and storing in array c.
    for (int i = 0; i < r1; ++i)
        for (int j = 0; j < c2; ++j)
            for (int k = 0; k < c1; ++k)
                c[i][j] += a[i][k] * b[k][j];

    return c;
}

double **mat_add(double **(&a), double **(&b), int r, int c) {
    double **add;
    init_2D_mat(add, r, c);
    // Multiplying matrix a and b and storing in array c.
    for (int i = 0; i < r; ++i)
        for (int j = 0; j < c; ++j)
            add[i][j] = a[i][j] + b[i][j];

    return add;
}

double **mat_transpose(double **a, int r, int c) {
    double **trans;
    init_2D_mat(trans, c, r);
    for (int i = 0; i < r; ++i)
        for (int j = 0; j < c; ++j)
            trans[j][i] = a[i][j];
    return trans;
}

double **sigmoid(double **mat, int r, int c) {
    double **s;
    init_2D_mat(s, r, c);
    // cout << "here " <<  double(1 /( 1 + exp(-mat[0][0]) )) << endl;
    
    for(unsigned i = 0; i < r; ++i) {
        
        for(unsigned j = 0; j < c; ++j) {
            s[i][j] = 1 / (1 + exp(-mat[i][j]));
        }
        
    }
    return s;
}

double *vector_add(double *(&a), double *(&b), int r) {
    double *add;
    init_1D_mat(add, r);
    // Multiplying matrix a and b and storing in array c.
    for (int i = 0; i < r; ++i)
        add[i] = a[i] + b[i];

    return add;
}

double **scalar_add_2D_mat(double **mat, int scalar, int r, int c) {
    for (unsigned i = 0; i < r; ++i)
    {
        for (unsigned j = 0; j < c; ++j)
        {
            mat[i][j] += scalar;
        }
    }
    return mat;
}

double **scalar_multiply_2D_mat(double **mat, int scalar, int r, int c) {
    for (unsigned i = 0; i < r; ++i)
    {
        for (unsigned j = 0; j < c; ++j)
        {
            mat[i][j] *= scalar;
        }
    }
    return mat;
}

double *scalar_multiply_1D_mat(double *mat, int scalar, int r) {
    for (unsigned i = 0; i < r; ++i)
    {
        mat[i] *= scalar;
    }
    return mat;
}

double **scalar_divide_2D_mat(double **mat, double scalar, int r, int c) {
    for (unsigned i = 0; i < r; ++i)
    {
        for (unsigned j = 0; j < c; ++j)
        {
            mat[i][j] /= scalar;
        }
    }
    return mat;
}

double *scalar_divide_1D_mat(double *mat, int scalar, int r) {
    for (unsigned i = 0; i < r; ++i)
    {
        mat[i] /= scalar;
    }
    return mat;
}

double **diff_2D_mat_1D_mat(double **a, double *b, int r, int c) {
    for (unsigned i = 0; i < r; ++i)
        for (unsigned j = 0; j < c; ++j)
            a[i][j] -= b[i];
    return a;
}

double **add_2D_mat_1D_mat(double **a, double *b, int r, int c) {
    for (unsigned i = 0; i < r; ++i)
        for (unsigned j = 0; j < c; ++j)
            a[i][j] += b[j];
    return a;
}

double **element_wise_multiply(double **a, double **b, int r, int c) {
    for (unsigned i = 0; i < r; ++i)
        for (unsigned j = 0; j < c; ++j)
            a[i][j] *= b[i][j];
    return a;
}

double *sum_across_2nd_dim(double **a, int r, int c) {
    double *sum;
    init_1D_mat(sum, r);
    for (unsigned i = 0; i < r; ++i) {
        int s = 0;
        for (unsigned j = 0; j < c; ++j)
            s += a[i][j];
        sum[i] = s;
    }
    return sum;
}

/********************************************************************/

double **dsigmoid(double **a, int r, int c) {
    double **sigmoid_a = sigmoid(a, r, c);
    double **one_minus_sigmoid_a;
    init_2D_mat(one_minus_sigmoid_a, r, c);
    for(int i = 0; i < r; i++)
	for(int j = 0; j < c; j++)
		one_minus_sigmoid_a[i][j] = 1 - sigmoid_a[i][j];
    double **d_sigmoid = cu_mat_elementwise_multiply_helper(sigmoid_a, one_minus_sigmoid_a, r, c);
    return d_sigmoid;
}

void forward_prop(double **(&X), double **(&W1), double **(&W2), double *(&b1), double *(&b2),
                      double **(&Z1), double **(&Z2), double **(&A1), double **(&A2), int examples) {
    
    //double **W1_trans = mat_transpose(W1, HIDDEN_LAYER_NODES, DATA_DIM);
    double **W1_trans = cuda_mat_transpose_helper(W1, HIDDEN_LAYER_NODES, DATA_DIM);
    // Z1 = mat_multiply(X, W1_trans, examples, DATA_DIM, DATA_DIM, HIDDEN_LAYER_NODES);
    Z1 = cuda_mat_multiply_helper(X, W1_trans, examples, DATA_DIM, DATA_DIM, HIDDEN_LAYER_NODES);
    A1 = sigmoid(Z1, examples, HIDDEN_LAYER_NODES);
    //A1 = cu_sigmoid_helper(Z1, examples, HIDDEN_LAYER_NODES);
    A1 = add_2D_mat_1D_mat(A1, b1, examples, HIDDEN_LAYER_NODES);
    //A1 = cu_2D_1D_addition_helper(A1, b1, examples, HIDDEN_LAYER_NODES);
 

    //double **W2_trans = mat_transpose(W2, OUTPUT_LAYER_NODES, HIDDEN_LAYER_NODES);
    double **W2_trans = cuda_mat_transpose_helper(W2, OUTPUT_LAYER_NODES, HIDDEN_LAYER_NODES);
    // Z2 = mat_multiply(A1, W2_trans, examples, HIDDEN_LAYER_NODES, HIDDEN_LAYER_NODES, OUTPUT_LAYER_NODES);
    Z2 = cuda_mat_multiply_helper(A1, W2_trans, examples, HIDDEN_LAYER_NODES, HIDDEN_LAYER_NODES, OUTPUT_LAYER_NODES);
    A2 = sigmoid(Z2, examples, OUTPUT_LAYER_NODES);
    //A2 = cu_sigmoid_helper(Z2, examples, OUTPUT_LAYER_NODES);
    A2 = add_2D_mat_1D_mat(A2, b2, examples, OUTPUT_LAYER_NODES);
    //A2 = cu_2D_1D_addition_helper(A2, b2, examples, OUTPUT_LAYER_NODES);

}

/*
void calculate_cost(double **(&A2), double *(&Y)) {
    double cost = 0;
    double **c = diff_2D_mat_1D_mat(A2, Y, TRAIN_SAMPLE, OUTPUT_LAYER_NODES);
    c = element_wise_multiply(c, c, TRAIN_SAMPLE, OUTPUT_LAYER_NODES);
    c = scalar_divide_2D_mat(c, TRAIN_SAMPLE, TRAIN_SAMPLE, OUTPUT_LAYER_NODES);
    for(unsigned i = 0; i < TRAIN_SAMPLE; ++i) {

        for (unsigned j = 0; j < OUTPUT_LAYER_NODES; ++j)
        {
            cost += c[i][j];
        }
        
    }
    printf("Cost: %0.9f\n", cost);
}
*/

void back_prop(double **(&X), double *(&Y), double **(&W1), double **(&W2),
               double **(&A1), double **(&A2), double **(&dW1), double **(&dW2),
               double **(&dA1), double **(&dA2), double *(&db1), double *(&db2)) {

    /*
    double **one;
    init_2D_mat(one, TRAIN_SAMPLE, HIDDEN_LAYER_NODES);
    // Initialize one with all 1
    for (unsigned i = 0; i < TRAIN_SAMPLE; ++i)
        for (unsigned j = 0; j < HIDDEN_LAYER_NODES; ++j)
            one[i][j] = 1;
    */
    
    double **dZ2 = diff_2D_mat_1D_mat(A2, Y, TRAIN_SAMPLE, OUTPUT_LAYER_NODES);
    
    //double **dZ2_trans = mat_transpose(dZ2, TRAIN_SAMPLE, OUTPUT_LAYER_NODES);
    double **dZ2_trans = cuda_mat_transpose_helper(dZ2, TRAIN_SAMPLE, OUTPUT_LAYER_NODES);
    
    // dW2 = mat_multiply(dZ2, A1, OUTPUT_LAYER_NODES, TRAIN_SAMPLE, TRAIN_SAMPLE, HIDDEN_LAYER_NODES);
    dW2 = cuda_mat_multiply_helper(dZ2, A1, OUTPUT_LAYER_NODES, TRAIN_SAMPLE, TRAIN_SAMPLE, HIDDEN_LAYER_NODES);
    
    //dW2 = scalar_divide_2D_mat(dW2, TRAIN_SAMPLE, OUTPUT_LAYER_NODES, HIDDEN_LAYER_NODES);
    dW2 = cu_mat_scalar_multiply_helper(dW2, 1/TRAIN_SAMPLE, OUTPUT_LAYER_NODES, HIDDEN_LAYER_NODES);
    
    db2 = sum_across_2nd_dim(dZ2_trans, OUTPUT_LAYER_NODES, TRAIN_SAMPLE);
    //db2 = scalar_divide_1D_mat(db2, TRAIN_SAMPLE, OUTPUT_LAYER_NODES);
    db2 = cu_vec_scalar_multiply_helper(db2, 1/TRAIN_SAMPLE, OUTPUT_LAYER_NODES);





    //double **A1_square = element_wise_multiply(A1, A1, TRAIN_SAMPLE, HIDDEN_LAYER_NODES);
    double **A1_square = cu_mat_elementwise_multiply_helper(A1, A1, TRAIN_SAMPLE, HIDDEN_LAYER_NODES);
    
    //A1_square = scalar_multiply_2D_mat(A1_square, -1, TRAIN_SAMPLE, HIDDEN_LAYER_NODES);
    A1_square = cu_mat_scalar_multiply_helper(A1_square, -1, TRAIN_SAMPLE, HIDDEN_LAYER_NODES);
    
    //A1_square = mat_add(one, A1_square, TRAIN_SAMPLE, HIDDEN_LAYER_NODES);
    A1_square = cu_addition_helper(one, A1_square, TRAIN_SAMPLE, HIDDEN_LAYER_NODES);
    
    //double **W2xdZ2 = mat_multiply(dZ2, W2, TRAIN_SAMPLE, OUTPUT_LAYER_NODES, OUTPUT_LAYER_NODES, HIDDEN_LAYER_NODES);
    double **W2xdZ2 = cuda_mat_multiply_helper(dZ2, W2, TRAIN_SAMPLE, OUTPUT_LAYER_NODES, OUTPUT_LAYER_NODES, HIDDEN_LAYER_NODES);
    
    //double **derivative_Z1 = cu_dsigmoid_helper(A1, TRAIN_SAMPLE, HIDDEN_LAYER_NODES);
    double **derivative_Z1 = dsigmoid(A1, TRAIN_SAMPLE, HIDDEN_LAYER_NODES);

    //double **dZ1 = element_wise_multiply(A1_square, W2xdZ2, TRAIN_SAMPLE, HIDDEN_LAYER_NODES);
    double **dZ1 = cu_mat_elementwise_multiply_helper(derivative_Z1, W2xdZ2, TRAIN_SAMPLE, HIDDEN_LAYER_NODES);
    
    //double **dZ1_trans = mat_transpose(dZ1, TRAIN_SAMPLE, HIDDEN_LAYER_NODES);
    double **dZ1_trans = cuda_mat_transpose_helper(dZ1, TRAIN_SAMPLE, HIDDEN_LAYER_NODES);
    
    // dW1 = mat_multiply(dZ1_trans, X, HIDDEN_LAYER_NODES, TRAIN_SAMPLE, TRAIN_SAMPLE, DATA_DIM);
    dW1 = cuda_mat_multiply_helper(dZ1_trans, X, HIDDEN_LAYER_NODES, TRAIN_SAMPLE, TRAIN_SAMPLE, DATA_DIM);
    
    //dW1 = scalar_divide_2D_mat(dW1, TRAIN_SAMPLE, HIDDEN_LAYER_NODES, DATA_DIM);
    dW1 = cu_mat_scalar_multiply_helper(dW1, 1/TRAIN_SAMPLE, HIDDEN_LAYER_NODES, DATA_DIM);

    db1 = sum_across_2nd_dim(dZ1_trans, HIDDEN_LAYER_NODES, TRAIN_SAMPLE);
    //db1 = scalar_divide_1D_mat(db1, TRAIN_SAMPLE, HIDDEN_LAYER_NODES);
    db1 = cu_vec_scalar_multiply_helper(db1, 1/TRAIN_SAMPLE, HIDDEN_LAYER_NODES);
}

void update_parameter(double **(&W1), double **(&W2), double *(&b1), double *(&b2),
                      double **(&dW1), double **(&dW2), double *(&db1), double *(&db2)) {


    //dW1 = scalar_multiply_2D_mat(dW1, (-1 * LEARNING_RATE_IP_HIDDEN), HIDDEN_LAYER_NODES, DATA_DIM);
    //dW2 = scalar_multiply_2D_mat(dW2, (-1 * LEARNING_RATE_HIDDEN_OP), OUTPUT_LAYER_NODES, HIDDEN_LAYER_NODES);
    //W1 = mat_add(W1, dW1, HIDDEN_LAYER_NODES, DATA_DIM);
    //W2 = mat_add(W2, dW2, OUTPUT_LAYER_NODES, HIDDEN_LAYER_NODES);
    
    dW1 = cu_mat_scalar_multiply_helper(dW1, (-1 * LEARNING_RATE_IP_HIDDEN), HIDDEN_LAYER_NODES, DATA_DIM);
    dW2 = cu_mat_scalar_multiply_helper(dW2, (-1 * LEARNING_RATE_HIDDEN_OP), OUTPUT_LAYER_NODES, HIDDEN_LAYER_NODES);
    W1 = cu_addition_helper(W1, dW1, HIDDEN_LAYER_NODES, DATA_DIM);
    W2 = cu_addition_helper (W2, dW2, OUTPUT_LAYER_NODES, HIDDEN_LAYER_NODES);
    
    //db1 = scalar_multiply_1D_mat(db1, (-1 * LEARNING_RATE_IP_HIDDEN), HIDDEN_LAYER_NODES);
    //db2 = scalar_multiply_1D_mat(db2, (-1 * LEARNING_RATE_HIDDEN_OP), OUTPUT_LAYER_NODES);
    //b1 = vector_add(b1, db1, HIDDEN_LAYER_NODES);
    //b2 = vector_add(b2, db2, OUTPUT_LAYER_NODES);
    
    db1 = cu_vec_scalar_multiply_helper(db1, (-1 * LEARNING_RATE_IP_HIDDEN), HIDDEN_LAYER_NODES);
    db2 = cu_vec_scalar_multiply_helper(db2, (-1 * LEARNING_RATE_HIDDEN_OP), OUTPUT_LAYER_NODES);
    b1 = cu_vec_addition_helper(b1, db1, HIDDEN_LAYER_NODES);
    b2 = cu_vec_addition_helper(b2, db2, OUTPUT_LAYER_NODES);

}

int main() {
    // Initialize dataset    
    double **train_x, **test_x, *train_y, *test_y;

    init_2D_mat(train_x, TRAIN_SAMPLE, DATA_DIM);
    init_2D_mat(test_x, TEST_SAMPLE, DATA_DIM);
    init_1D_mat(train_y, TRAIN_SAMPLE);
    init_1D_mat(test_y, TEST_SAMPLE);

    // load train data
    load_data(train_x, train_y, TRAIN_SAMPLE, DATA_DIM + 1, (const char*)"data_500.txt");

    // print2Dmat(train_x, TRAIN_SAMPLE, DATA_DIM);
    // print1Dmat(train_y, TRAIN_SAMPLE);

    // load test data
    load_data(test_x, test_y, TEST_SAMPLE, DATA_DIM + 1, (const char *)"test_data_121.txt");


    // Initialize FW prop weights
    double **W1, **W2, *b1, *b2;
    init_2D_mat(W1, HIDDEN_LAYER_NODES, DATA_DIM);
    init_2D_mat(W2, OUTPUT_LAYER_NODES, HIDDEN_LAYER_NODES);

    init_1D_mat(b1, HIDDEN_LAYER_NODES);
    init_1D_mat(b2, OUTPUT_LAYER_NODES);

    init_weights(W1, HIDDEN_LAYER_NODES, DATA_DIM);
    init_weights(W2, OUTPUT_LAYER_NODES, HIDDEN_LAYER_NODES);

    init_biases(b1, HIDDEN_LAYER_NODES);
    init_biases(b2, OUTPUT_LAYER_NODES);

    // Initialize Back Prop delta
    double **dW1, **dW2, *db1, *db2;
    init_2D_mat(dW1, HIDDEN_LAYER_NODES, DATA_DIM);
    init_2D_mat(dW2, OUTPUT_LAYER_NODES, HIDDEN_LAYER_NODES);
    init_1D_mat(db1, HIDDEN_LAYER_NODES);
    init_1D_mat(db2, OUTPUT_LAYER_NODES);

    // Initiaze FW prop return parameters
    double **Z1, **Z2, **A1, **A2;
    init_2D_mat(Z1, TRAIN_SAMPLE, HIDDEN_LAYER_NODES);
    init_2D_mat(A1, TRAIN_SAMPLE, HIDDEN_LAYER_NODES);
    init_2D_mat(Z2, TRAIN_SAMPLE, OUTPUT_LAYER_NODES);
    init_2D_mat(A2, TRAIN_SAMPLE, OUTPUT_LAYER_NODES);

    // Initiaze Back prop return parameters
    double **dA1, **dA2;
    init_2D_mat(dA1, TRAIN_SAMPLE, HIDDEN_LAYER_NODES);
    init_2D_mat(dA2, TRAIN_SAMPLE, OUTPUT_LAYER_NODES);

    // Train neural network
    for(unsigned i = 0; i < EPOCH; ++i) {
        cout << "------Iteration: " << i+1 << endl;
        // Forward Propagation
        forward_prop(train_x, W1, W2, b1, b2, Z1, Z2, A1, A2, TRAIN_SAMPLE);

        // Calculate cost
        // calculate_cost(A2, train_y);

        // Backward Propagation
        back_prop(train_x, train_y, W1, W2, A1, A2, dW1, dW2, dW1, dW2, db1, db2);

        
        // Parameter Updates
        update_parameter(W1, W2, b1, b2, dW1, dW2, db1, db2);
    }
    //print2Dmat(W2, OUTPUT_LAYER_NODES, HIDDEN_LAYER_NODES);
    // Initiaze FW prop return parameters for test
    double **_Z1, **_Z2, **_A1, **_A2;
    init_2D_mat(_Z1, TEST_SAMPLE, HIDDEN_LAYER_NODES);
    init_2D_mat(_A1, TEST_SAMPLE, HIDDEN_LAYER_NODES);
    init_2D_mat(_Z2, TEST_SAMPLE, OUTPUT_LAYER_NODES);
    init_2D_mat(_A2, TEST_SAMPLE, OUTPUT_LAYER_NODES);

    // Test the network
    forward_prop(test_x, W1, W2, b1, b2, _Z1, _Z2, _A1, _A2, TEST_SAMPLE);

    for(unsigned i = 0; i < TEST_SAMPLE; ++i) {
        for (unsigned j = 0; j < OUTPUT_LAYER_NODES; ++j)
            if (A2[i][j] >= 0.5) A2[i][j] = 1;
            else A2[i][j] = 0;
    }
    int accurate = 0;
    for (unsigned i = 0; i < TEST_SAMPLE; ++i) {
        for (unsigned j = 0; j < OUTPUT_LAYER_NODES; ++j) {
            //cout << A2[i][j] << " ";
            if (A2[i][j] == test_y[i])
                accurate++;
        }
    }
    cout << "\n\nAccuracy of the model is " << ((accurate * 100) / TEST_SAMPLE) << endl;

    return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <immintrin.h>
#include <math.h>


#define MAX_BASE_DIMENSION  200
#define MAX_TARGET_DIMENSION 64
#define MAX_DISTANCE_VECTOR_DIMENSION 64
#define RAND 1
#define PRINT_DIGITALS 32


/*
This program demonstrates the three approaches of "Step 4" of the report "Efficient implementations of BP heuristic for the Shortest
Linear Program (SLP) Problem".

In order to run this code:
1. Make sure the input file follow this format:
    row col
    [matrix]

    for example:
    3 4
    1 0 1 1
    0 1 0 1
    1 1 1 1

2. Change the Macro "ROW" and "COL" to match with the input martix.
3. Specify the location of the file by using inputfilename[] or fileTemplate[].
    The program provides two modes:
    ONE_MATRIX_MODE provides an evaluation of a single matrix. The file name looks like "..\\AES.txt"
    MULTI_MATRIX_MODE provides evaluations of 20 matrices (with the same size) under the same folder.  The template name looks like "..\\folderName\\matrix_%d.txt"
4. Select either option: NAIVE, SIMD or BINARY
5. Select how many rounds of BP with randomization by changing the MACRO "ROUNDS_FOR_RAND".



For debugging, the program provides two additional options:
PRINT_DETAIL allows the program to print out details for each iteration.
CHECK_CIRCUIT allows the program to check the resulted circuit by generating 50 random inputs and verify the output.
*/


            //Matrix size
            #define ROW 32
            #define COL 32

            //Matrix file location
            #define ONE_MATRIX_MODE 1
            char inputfilename[] = ".\\benchmarkMatrix\\AES.txt";

            #define MUTI_MATRIX_MODE 0
            char fileTemplate[] = ".\\benchmarkMatrix\\matrices_20x20_0.2\\matrix_%d.txt";

            //Naive
            #define NAIVE 0

            //SIMD
            #define SIMD 0

            //Binary search
            #define BINARY 1

            //RANDOMIZATION
            #define ROUNDS_FOR_RAND 5

            //Debug Options
            #define PRINT_DETAIL 0      //Print details for each iteration
            #define PRINT_NUM_OF_BASE 100 //Choose how many bases to be displayed
            #define CHECK_CIRCUIT 1     //Check if the result circuit is correct


//Functions for initialization
void readMatrix(size_t rows, size_t cols, int (*a)[cols], char (*inputfilename));                                  //Read matrix from file
void toUint32(int row, int col, int (*mat)[col], uint32_t (*arr));                                               //Save the each row array into an uint32_t scalar
int initializeDistVectorTarget();

//Functions for "pre-emptive" choice
int isIn(int a, int rows, int cols, int *mat_pointer);                                                             //Check if any target distance is 1
int canReachOne(uint32_t target, int deepth, int x, const uint32_t new_base);                                   //Check if can reach "target" with "deepth" number of base in base_arr[0...x]

//Functions for Naive approach
int updateDistMaskNaive(int count, int temp_max_index, uint32_t result);                                           //Update distance masks in M

//Functions for SIMD approach
int laneIndex(uint32_t mask);                                                                                       //Return the index of the matched lane
int updateDistMaskSimd(int count, int temp_max_index, uint32_t result, const int (*nu),const int iota);         //Update distance masks in M

//Functions for Binary search approach
int updateDistMaskBinary(int count, int temp_max_index, uint32_t result);                                          //Update distance masks in M_sorted
void mergeSort(uint32_t arr[][MAX_BASE_DIMENSION*MAX_BASE_DIMENSION], int l, int r);
void merge(uint32_t arr[][MAX_BASE_DIMENSION*MAX_BASE_DIMENSION], int l, int m, int r);


//Functions for process distance masks and determine the best choice of new base
void findBestNewBase();                                                                                              //Find the best choice of new base according to M
void findBestNewBaseSorted();                                                                                        //Find the best choice of new base according to M_sorted
double normSquare(int num_of_target, int dist_vector[]);                                                            //Return the square of norm of the distance vector


//Other functions
int initializeMatrix(int rows, int cols, int(*a)[cols],int value);
int initializeMatrixPartial(int cols, uint32_t (*a)[cols], int prows, int pcols, int value);
int distanceSum();                                                                                                     //Return the sum of a distance vector
int check();                                                                                                           //Check if the result circuit is correct
void stopWatch(int flag);                                                                                              //Timer


//Global variables declaration
int row = ROW, col = COL;
int num_of_target;
int num_of_base;
int num_of_new_base;                                                        //Number of new bases


int         target_matrix[ROW][COL];                                        //Store each target as a row array
uint32_t    target_arr[MAX_TARGET_DIMENSION];                               //Store each target as a 32-bit scalar

int         base_matrix[MAX_BASE_DIMENSION][COL];                           //Store each base as a row in matrix
uint32_t    base_arr[MAX_BASE_DIMENSION];                                   //Store each base as a 32-bit scalar

uint32_t   __attribute__ ((aligned(32))) M[67][MAX_BASE_DIMENSION*MAX_BASE_DIMENSION];          //Check Section 5.3 for defination
uint32_t    __attribute__ ((aligned(32))) M_sorted[67][MAX_BASE_DIMENSION*MAX_BASE_DIMENSION];  //Check Section 5.3 for defination
uint32_t    __attribute__ ((aligned(32))) M_raw[67][MAX_BASE_DIMENSION*MAX_BASE_DIMENSION];     //Check Section 5.3 for defination

int         dist_vector[MAX_DISTANCE_VECTOR_DIMENSION];                 //Distance vector
int         best_new_base_index;                                        //The index for the best column in M or M_sorted
int         best_pair[2];                                               /*The best pair of bases for the iteration (M[65][best_new_base_index], M[66][best_new_base_index])
                                                                            or M_sorted[65][best_new_base_index], M_sorted[66][best_new_base_index])*/

int         iota;                                                           //Number of target with the same distance
int         nu[MAX_TARGET_DIMENSION];                                       //Index of target with the same distance
int         circuit[MAX_BASE_DIMENSION][4];                                 //Resulted circuit


//Declare SIMD related variables
uint32_t    __attribute__ ((aligned(32))) target_arr8[MAX_TARGET_DIMENSION*8];       //Align targets to be loaded into SIMD register
int         num_of_new_base8vec;                                                     //Number of new bases vector in terms of 8 lanes

int main(void){

#if ONE_MATRIX_MODE
//Setup timer
clock_t start_time, end_time;
double time_used;

//Start timer
start_time = clock();

int best_count = 100000;
int temp_count, best_round;


for(size_t i = 0; i < ROUNDS_FOR_RAND; i++)
{
    printf("Round: %d\n", i);
    printf("%s\n", inputfilename);

    temp_count = evaluateMatrix(inputfilename);

    if (temp_count < best_count)
    {
        best_count = temp_count;
        best_round = i;
    }
}

printf("At round %d, have best circuit with %d XOR\n", best_round, best_count);

//End timer, print total time
end_time = clock();
time_used = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
printf("Average timing is %f\n", time_used/(ROUNDS_FOR_RAND));


#endif // ONE matrix mode

#if MUTI_MATRIX_MODE
char filename[100];

//Setup total timer
clock_t total_start_time, total_end_time;
double total_time_used;

//Start total timer
total_start_time = clock();

 for(size_t file_count = 0; file_count < 20; ++ file_count ){

    //Setup individual timer
    clock_t start_time, end_time;
    double time_used;

    //Start individual timer
    start_time = clock();

    //Generate the filename
    snprintf(filename, 100, fileTemplate, file_count);
    printf("%s\n", filename);

    int best_count = 100000;
    int temp_count, best_round;
    for(size_t i = 0; i < ROUNDS_FOR_RAND; i++)
    {
        temp_count = evaluateMatrix(filename);

        if (temp_count < best_count)
        {
            best_count = temp_count;
            best_round = i;
        }
    }

    printf("At round %d, have best circuit with %d XOR\n", best_round, best_count);

    //End individual timer, print total time
    end_time = clock();
    time_used = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
    printf("Running time %f for matrix %d\n", time_used, file_count);

 }

//End total timer, print total time
total_end_time = clock();
total_time_used = ((double)(total_end_time - total_start_time)) / CLOCKS_PER_SEC;
printf("Average timing is %f\n", total_time_used/(ROUNDS_FOR_RAND*20));

#endif // MUTI_MATRIX_MODE

system("pause");
}

int evaluateMatrix(char (*inputfilename)){  //Evaluate matrix, return the number of XOR needed

    //Open file and read in matrix row and column
    FILE *fp;
    fp = fopen(inputfilename,"rt");
    if (fp == NULL){
        printf("Could not open file %s",inputfilename);
        return 1;
    }
    int actual_row, actual_col;
    fscanf(fp, "%d" ,&actual_row);
    fscanf(fp, "%d", &actual_col);
    fclose(fp);

    if((actual_col != col) || (actual_row != row))
    {
        printf("Size not match\n");
        exit(-1);
    }

    if(col >32)
    {
        printf("Matrix oversized\n");
        exit(-1);
    }
    num_of_target = row;
    num_of_base = col;

    //Read in target matrix
    readMatrix(row, col, target_matrix, inputfilename);


    //Initialize base matrix to each input variable
    initializeMatrix(MAX_BASE_DIMENSION, col, base_matrix, 0);
    for(size_t i = 0; i < col; ++i)
    {
        base_matrix[i][i] = 1;
    }

    //Load the targets and bases into uint_32 data structure
    toUint32(MAX_BASE_DIMENSION,col, base_matrix,base_arr);
    toUint32(row, col, target_matrix, target_arr);




    //Initialize distance vector
    initializeDistVectorTarget();

    //Initialize circuit
    initializeMatrix(MAX_BASE_DIMENSION, 4, circuit, -1);

    #if SIMD
    //Align the targets to be broadcast
    for(size_t i = 0; i < num_of_target; i++)
    {
        for(size_t j = 0; j < 8; j++)
        {
            target_arr8[i*8+j] = target_arr[i];
        }
    }
    #endif // SIMD


    #if (PRINT_DETAIL == 1)


    printf("Target matrix:\n");
    printUint32Arr(row,target_arr);
    printf("Base matrix:\n");
    printUint32Arr(PRINT_NUM_OF_BASE, base_arr);

    #endif // print_detail

    while(distanceSum()!=0)//when sum of distance not zero
    {

        #if (PRINT_DETAIL ==1)
        printf("Iteration %d\n", num_of_base - col + 1);
        #endif // PRINT_DETAIL

        new_base();

        //Print out the new base and save to circuit[]
        printf("t%d = b%d + b%d\n",num_of_base - col, best_pair[0], best_pair[1]);
        circuit[num_of_base][0]= best_pair[0];
        circuit[num_of_base][1]= best_pair[1];

        num_of_base++;

        #if (PRINT_DETAIL == 1)

        #if BINARY
         printf("M_sorted:\n");
        for(size_t c = 0; c < num_of_new_base; c++)
        {
            //Print the indexes of old bases
            printf("b%2d + b%2d     ", M_sorted[65][c], M_sorted[66][c]);

            //Print the new base
            unsigned long long int temp = M_sorted[0][c];
            uint32_t reversed_num = 0;
            //reverse the number so can print from high digit to low digit
            for(size_t c = 0; c < PRINT_DIGITALS; ++c )
            {
                reversed_num = temp%2 + reversed_num * 2;
                temp = temp/2;
            }

            for(size_t i = 0; i< PRINT_DIGITALS; ++i)
            {
                int a = reversed_num%2;
                printf("%d", a);
                reversed_num = reversed_num/2;
            }

            printf("        ");
            //Print distance mask
            for(size_t r = 1; r < COL+1; r++)
            {
                printf("%d ", M_sorted[r][c]);
            }

            printf("\n");
        }


        #else
        printf("M:\n");
        for(size_t c = 0; c < num_of_new_base; c++)
        {
            //Print the indexes of old bases
            printf("b%2d + b%2d     ", M[65][c], M[66][c]);

            //Print the new base
            unsigned long long int temp = M[0][c];
            uint32_t reversed_num = 0;
            //reverse the number so can print from high digit to low digit
            for(size_t c = 0; c < PRINT_DIGITALS; ++c )
            {
                reversed_num = temp%2 + reversed_num * 2;
                temp = temp/2;
            }

            for(size_t i = 0; i< PRINT_DIGITALS; ++i)
            {
                int a = reversed_num%2;
                printf("%d", a);
                reversed_num = reversed_num/2;
            }

            printf("        ");
            //Print distance mask
            for(size_t r = 1; r < COL+1; r++)
            {
                printf("%d ", M[r][c]);
            }

            printf("\n");
        }


        #endif // BINARY


        printf("New base matrix:\n");
        printUint32Arr(PRINT_NUM_OF_BASE, base_arr);
        printf("Distance vector: ");
        printMatrix(1,num_of_target,dist_vector);
        printf("\n");
        #endif // PRINT_DETAIL


    }

    #if CHECK_CIRCUIT
    //Check if the result circuit is correct
    for(size_t i = 0; i < 50; i++)
    {
        if(!check()){
                printf("Circuit incorrect.\n");
                system("pause");
        }
    }
    printf("Circuit passes 50 rounds of random input checks.\n");
    #endif // CHECK_CIRCUIT



    int total_number_of_XOR = num_of_base - col;

    return total_number_of_XOR;
}

int new_base(){ //Choose new base for the iteration


    int num_of_temp_new_base = (num_of_base * num_of_base - num_of_base)/2; //Number of new base before removing duplicates

    //Initialize meanful entries of M_raw, M or M_sorted to 0
    #if BINARY
    initializeMatrixPartial(MAX_BASE_DIMENSION*MAX_BASE_DIMENSION, M_raw, num_of_target+3, num_of_temp_new_base+1,0);
    initializeMatrixPartial(MAX_BASE_DIMENSION*MAX_BASE_DIMENSION, M_sorted, num_of_target+3, num_of_temp_new_base+1,0);
    #else
    initializeMatrixPartial(MAX_BASE_DIMENSION*MAX_BASE_DIMENSION, M_raw, num_of_target+3, num_of_temp_new_base+1,0);
    initializeMatrixPartial(MAX_BASE_DIMENSION*MAX_BASE_DIMENSION, M, num_of_target+1, num_of_temp_new_base+1,0);
    #endif // BINARY



    //Set M_raw
    int temp = 0;
    for(size_t r = 0; r < num_of_base; ++r)
    {
        for(size_t c= r+1; c< num_of_base;++c)
        {
            M_raw[0][temp]= base_arr[r] ^ base_arr[c];
            M_raw[65][temp] = r;
            M_raw[66][temp] = c;
            temp++;
        }
    }


     #if BINARY
        //Sort the new bases
        mergeSort(M_raw, 0, num_of_temp_new_base - 1);


        //Remove duplications, based on "https://www.studytonight.com/c/programs/array/remove-duplicate-element-program"
          int current_index = 0;
          for (size_t i = 0; i < num_of_temp_new_base - 1; i++){
            if (M_raw[0][i] != M_raw[0][i + 1])
                {
                    M_sorted[0][current_index] = M_raw[0][i];
                    M_sorted[65][current_index] = M_raw[65][i];
                    M_sorted[66][current_index] = M_raw[66][i];
                    current_index++;
                }
          }

        M_sorted[0][current_index] = M_raw[0][num_of_temp_new_base-1];
        M_sorted[65][current_index] = M_raw[65][num_of_temp_new_base-1];
        M_sorted[66][current_index] = M_raw[66][num_of_temp_new_base-1];
        num_of_new_base = current_index+1;
     #else
        //Remove repetitions, based on "https://www.studytonight.com/c/programs/array/remove-duplicate-element-program"
        int n=num_of_temp_new_base, count = 0;

          for (int i = 0; i < n; i++)
          {
            int key;
            for (key = 0; key < count; key++)
            {
              if (M_raw[0][i] == M[0][key])
                break;

            }
            if (key == count)
            {
              M[0][count] = M_raw[0][i];
              M[65][count] = M_raw[65][i];
              M[66][count] = M_raw[66][i];
              count++;
            }
          }

        for(size_t i = 0; i < num_of_temp_new_base+1; i++)
        {
            if(M[0][i]==0)
            {
                num_of_new_base = i;
                break;
            }
        }

     #endif // BINARY



    #if SIMD
    if(num_of_new_base%8 != 0)
    {
        num_of_new_base8vec = num_of_new_base / 8 + 1;
    }
    else{
        num_of_new_base8vec = num_of_new_base / 8;
    }
    #endif // SIMD


    if(isIn(1,1,num_of_target,dist_vector)) //When there is target which is direct sum of two bases
    {
        //Find which target have distance 1
        int dist_one_index = -1;
        for(size_t i =0; i<num_of_target;i++)
        {
            if(dist_vector[i]==1)
            {
                dist_one_index = i;
                break;
            }
        }

        #if BINARY
        // Update the circuit
        circuit[num_of_base][2] = target_arr[dist_one_index];

        //Find the new base gives this target
        int new_base_index=-1;
        for(size_t b = 0; b <num_of_new_base;++b)
        {
            if(M_sorted[0][b] == target_arr[dist_one_index])
            {

                new_base_index = b;
                break;
            }
        }

        //Update the distance mask
        uint32_t new_base = M_sorted[0][new_base_index];

        for (size_t t = 0; t < num_of_target;++t)
        {
            if(dist_vector[t] > 0){
                if(canReachOne(target_arr[t], dist_vector[t] - 1, num_of_base-1, new_base))
                {
                    M_sorted[t+1][new_base_index] = 1;
                }
            }
        }

        best_new_base_index = new_base_index;
        best_pair[0] = M_sorted[65][best_new_base_index];
        best_pair[1] = M_sorted[66][best_new_base_index];


        //Update dist_vector according to the best pair
        for(size_t i = 0; i<num_of_target;++i){
            dist_vector[i] = dist_vector[i] - M_sorted[i+1][best_new_base_index];
        }


        //Add the new base to the base set
        base_arr[num_of_base] = M_sorted[0][best_new_base_index];


        #else
        // Update the circuit
        circuit[num_of_base][2] = target_arr[dist_one_index];

        //Find the new base gives this target
        int new_base_index=-1;
        for(size_t b = 0; b <num_of_new_base;++b)
        {

            if(M[0][b] == target_arr[dist_one_index])
            {
                new_base_index = b;
                break;
            }
        }


        //Update the distance mask
        uint32_t new_base = M[0][new_base_index];

        for (size_t t = 0; t < num_of_target;++t)
        {
            if(dist_vector[t] > 0){

                if(canReachOne(target_arr[t], dist_vector[t] - 1, num_of_base-1, new_base))
                {
                    M[t+1][new_base_index] = 1;
                }
            }

        }

        best_new_base_index = new_base_index;
        best_pair[0] = M[65][best_new_base_index];
        best_pair[1] = M[66][best_new_base_index];

        //Update dist_vector according to the best pair
        for(size_t i = 0; i<num_of_target;++i){
            dist_vector[i] = dist_vector[i] - M[i+1][best_new_base_index];
        }

        //Add the new base to the base set
        base_arr[num_of_base] = M[0][best_new_base_index];
        #endif

        return 1; //Next iteration
    }
    else
    {

        for (size_t d = 1; d< col;++d) // "col" is the max distance a target could have
        {
            //Store the indexes of target with distance d
            iota = 0;
            for(size_t i = 0; i< num_of_target;++i)
            {
                if (dist_vector[i] == d) iota++;
            }
            if (iota == 0) continue;

            int index =0;
            for(size_t i = 0; i< num_of_target;++i)
            {
                if (dist_vector[i] == d)
                {
                    nu[index++] = i;
                }
            }

            //Print out the result
            #if PRINT_DETAIL == 1
            printf("For distance %d, have %d targets\n", d, iota);
            printf("  The index vector is:" );
            for(size_t i=0; i<iota;++i)
            {
                printf("%d ", nu[i]);
            }
            printf("\n");
            #endif // print_detail


            //Update distance masks
            #if SIMD
             updateDistMaskSimd(d-1, -1,  0, nu,iota);
            #endif

            #if BINARY
            updateDistMaskBinary(d-1, -1,  0);
            #endif // BINARY

            #if NAIVE
            updateDistMaskNaive(d-1,-1,0);
            #endif // BINARY
        }

         //Find the best new base, update best_pair[0], best_pair[1]
        #if BINARY
        findBestNewBaseSorted();
        #else
        findBestNewBase();
        #endif

    }






    return 0;
}


/*The following codes for the initialization iteration*/

void toUint32(int row, int col, int (*mat)[col], uint32_t (*arr)){ //The function save row vectors into uint32_t scalars

    //initialize arr to 0
    for(size_t c=0; c < row; ++c){

        arr[c]=0;
    }

    for(size_t r = 0; r < row; ++r)
    {
        for(size_t c = 0; c < col; ++c)
        {
            arr[r] = mat[r][c] + 2 * arr[r];
        }

    }

    return;

}

void readMatrix(size_t rows, size_t cols, int (*a)[cols], char (*inputfilename)){   //Read matrix from file

    FILE *fp;
    fp = fopen (inputfilename, "r");
    if (fp == NULL)
        return 0;

    int temp_rows, temp_cols;
    fscanf(fp, "%d" ,&temp_rows);
    fscanf(fp, "%d", &temp_cols);

    for(size_t i = 0; i < rows; ++i)
    {
        for(size_t j = 0; j < cols; ++j)
            fscanf(fp, "%d", a[i] + j);
    }

    fclose (fp);

}

int initializeDistVectorTarget(){

        //initialize distance to (hamming weight - 1)
        for(size_t i =0 ; i < num_of_target; ++i)
        {
            dist_vector[i] = -1;
            uint32_t temp = target_arr[i];

            for(size_t j = 0; j < col; ++j)
            {
                if(temp%2 == 1)dist_vector[i] ++;
                temp >>= 1;
            }
        }

        return 0;
}



/* The following code is for pre-emptive choice*/

int isIn(int a, int rows, int cols, int *mat_pointer){
    //Return if number "a" is in matrix

    int countNum = 0;

    for(size_t r =0; r<rows; ++r){
        for(size_t c = 0; c< cols; ++c){
            if(*(mat_pointer + r * cols + c )==a)countNum++;
        }
    }

    return countNum;

}

int canReachOne(uint32_t target, int deepth, int x, const uint32_t new_base){ // Return 1 if can reach "target" with "deepth" number of base in "base_arr[0...x]"

    if(deepth > x +1) return 0;

    if(deepth < 0) return 0;

    if(deepth == 0 && target == new_base) return 1;

    if(deepth == 0) return 0;

    if(deepth == 1) {
        for(size_t i = 0; i <= x; ++i){
            if(target== (new_base ^ base_arr[i]) ){
                return 1;
            }
        }

        return 0;
    }

    if(canReachOne(target^base_arr[x], deepth -1 , x-1, new_base)==1) return 1;

    if(canReachOne(target, deepth, x-1,  new_base)==1) return 1;

    return 0;
}


/*The following code is for NAIVE approaches*/

#if NAIVE
int updateDistMaskNaive(int count, int temp_max_index, uint32_t result){


    if(count < 0) return 0;
    else if(count == 0) //Reach a sum of count-combination
    {
        for(size_t i = 0; i < iota; ++i) //Go through each target with distance d
        {
                for(size_t index = 0; index < num_of_new_base; index++) //Go through each new base
                {
                    if((result^ M[0][index])== target_arr[nu[i]])
                    {
                        M[nu[i]+1][index] = 1; //Update distance mask
                        break;
                    }
                }

        }

        return 1;
    }
    else
    {
        for(size_t i = temp_max_index+1; i < num_of_base; ++i)
        {
            updateDistMaskNaive(count-1, i, result ^ base_arr[i]);
        }
        return 3;
    }

}

#endif // NAIVE


/*The following code is for SIMD process*/
#if SIMD
int laneIndex(uint32_t mask){
   //Input 32bit mask, 4bit each lane, output index of which lane is 0xf

   for(size_t i = 0; i < 8; ++i)
   {
       if(mask & 0xf) return i;
       mask = mask >> 4;
   }

   printf("\nError while converting index\n");
   system("pause");

   return 0;
}

int updateDistMaskSimd(int count, int temp_max_index, uint32_t result, const int (*nu),const int iota){

    if(count < 0) return 0;
    else if(count == 0) //Reach a sum of count-combination
    {

            for(size_t k = 0; k < iota; ++k) //Go through each target with distance d
            {

                __m256i simd_result = _mm256_xor_si256(_mm256_load_si256(&target_arr8[nu[k]*8]), _mm256_set1_epi32(result));

                for(size_t i = 0; i < num_of_new_base8vec; i++)
                {
                    int mask=_mm256_movemask_epi8(_mm256_cmpeq_epi32(simd_result, _mm256_load_si256(&M[0][i*8])));


                        if (mask!=0) //If a match is found
                        {
                            int index = laneIndex(mask) +i*8; //Convert the mask into index
                            M[nu[k]+1][index] = 1;
                            break;
                        }
                }
            }

        return 1;
    }
    else
    {
        for(size_t i = temp_max_index+1; i < num_of_base; ++i)
        {
            updateDistMaskSimd(count-1, i, result ^ base_arr[i],  nu, iota);
        }
        return 3;
    }

}

#endif // SIMD


/*The following code is for Binary search process*/
#if BINARY

void mergeSortR(uint32_t arr[][32], int l, int r){
    /* l is for left index and r is right index of the sub-array of arr to be sorted */

    //Sorting algorithm based on "https://www.geeksforgeeks.org/time-complexities-of-all-sorting-algorithms/"

    if (l < r) {

        int m = l + (r - l) / 2;// Same as (l+r)/2, but avoids overflow for large l and h

        // Sort first and second halves
        mergeSortR(arr, l, m);
        mergeSortR(arr, m + 1, r);
        mergeR(arr, l, m, r);
    }
}

void mergeR(uint32_t arr[][32], int l, int m, int r){

    int i, j, k;
    int n1 = m - l + 1;
    int n2 = r - m;

    /* create temp arrays */
    uint32_t L[3][n1], R[3][n2];

    /* Copy data to temp arrays L[] and R[] */
    for (i = 0; i < n1; i++)
    {
        L[0][i] = arr[0][l + i];
        L[1][i] = arr[1][l + i];

    }

    for (j = 0; j < n2; j++)
    {
        R[0][j] = arr[0][m + 1 + j];
        R[1][j] = arr[1][m + 1 + j];
    }

    /* Merge the temp arrays back into arr[l..r]*/
    i = 0; // Initial index of first subarray
    j = 0; // Initial index of second subarray
    k = l; // Initial index of merged subarray
    while (i < n1 && j < n2) {
        if (L[0][i] <= R[0][j]) {
            arr[0][k] = L[0][i];
            arr[1][k] = L[1][i];
            i++;
        }
        else {
            arr[0][k] = R[0][j];
            arr[1][k] = R[1][j];
            j++;
        }
        k++;
    }

    /* Copy the remaining elements of L[], if there
    are any */
    while (i < n1) {
        arr[0][k] = L[0][i];
        arr[1][k] = L[1][i];
        i++;
        k++;
    }

    /* Copy the remaining elements of R[], if there
    are any */
    while (j < n2) {
        arr[0][k] = R[0][j];
        arr[1][k] = R[1][j];
        j++;
        k++;
    }
}

int updateDistMaskBinary(int count, int temp_max_index, uint32_t result){


    if(count < 0) return 0;
    else if(count == 0) //Reach a sum of count-combination
    {
        int have = 0;
        #if 0
        int searchFlag = 0;
        uint32_t resultSort[2][32];

        for(size_t i = 0;i <iota;++i)
        {

            resultSort[0][i] = target_arr[nu[i]] ^ result;
            resultSort[1][i] = nu[i];
        }

        //sort resultSort[][]
        //mergeSortR(resultSort, 0, iota - 1);

        uint32_t tempI, tempII;
        for (size_t i = 0; i < iota; ++i)
        {

            for (size_t j = i + 1; j < iota; ++j)
            {

                if (resultSort[0][i] > resultSort[0][j])
                {

                    tempI =  resultSort[0][i];
                    tempII =   resultSort[1][i];
                    resultSort[0][i] = resultSort[0][j];
                    resultSort[1][i] = resultSort[1][j];
                    resultSort[0][j] = tempI;
                    resultSort[1][j] = tempII;

                }
            }
        }

        for(size_t i = 0; i < iota; ++i)//Go through each target with distance d, using binary search to find a matched new base
        {
            int first = searchFlag;
            int last = num_of_new_base - 1;
            int middle = (first+last)/2;
            uint32_t search = resultSort[0][i];

            while (first <= last)
            {
                if (M_sorted[0][middle] < search)
                  first = middle + 1;
                else if (M_sorted[0][middle] == search)
                {
                  M_sorted[resultSort[1][i]+1][middle] = 1;
                  searchFlag = middle+1;
                  break;
                }
                else last = middle - 1;

                middle = (first + last)/2;

            }

        }
        #else
        for(size_t i = 0; i < iota; ++i)//Go through each target with distance d, using binary search to find a matched new base
        {
            int first = 0;
            int last = num_of_new_base - 1;
            int middle = (first+last)/2;
            uint32_t search = target_arr[nu[i]] ^ result;

            while (first <= last)
            {
                if (M_sorted[0][middle] < search)
                  first = middle + 1;
                else if (M_sorted[0][middle] == search)
                {
                  M_sorted[nu[i]+1][middle] = 1;
                  //searchFlag = middle+1;
                  break;
                }
                else last = middle - 1;

                middle = (first + last)/2;

            }

        }
        #endif // 1


        return 1;
    }
    else
    {
        for(size_t i = temp_max_index+1; i < num_of_base; ++i)
        {
            updateDistMaskBinary(count-1, i, result ^ base_arr[i]);
        }
        return 3;
    }

}

void mergeSort(uint32_t arr[][MAX_BASE_DIMENSION*MAX_BASE_DIMENSION], int l, int r){
    /* l is for left index and r is right index of the sub-array of arr to be sorted */

    //Sorting algorithm based on "https://www.geeksforgeeks.org/time-complexities-of-all-sorting-algorithms/"

    if (l < r) {

        int m = l + (r - l) / 2;// Same as (l+r)/2, but avoids overflow for large l and h

        // Sort first and second halves
        mergeSort(arr, l, m);
        mergeSort(arr, m + 1, r);
        merge(arr, l, m, r);
    }
}

void merge(uint32_t arr[][MAX_BASE_DIMENSION*MAX_BASE_DIMENSION], int l, int m, int r){

    int i, j, k;
    int n1 = m - l + 1;
    int n2 = r - m;

    /* create temp arrays */
    uint32_t L[3][n1], R[3][n2];

    /* Copy data to temp arrays L[] and R[] */
    for (i = 0; i < n1; i++)
    {
        L[0][i] = arr[0][l + i];
        L[1][i] = arr[65][l + i];
        L[2][i] = arr[66][l + i];
    }

    for (j = 0; j < n2; j++)
    {
        R[0][j] = arr[0][m + 1 + j];
        R[1][j] = arr[65][m + 1 + j];
        R[2][j] = arr[66][m + 1 + j];
    }

    /* Merge the temp arrays back into arr[l..r]*/
    i = 0; // Initial index of first subarray
    j = 0; // Initial index of second subarray
    k = l; // Initial index of merged subarray
    while (i < n1 && j < n2) {
        if (L[0][i] <= R[0][j]) {
            arr[0][k] = L[0][i];
            arr[65][k] = L[1][i];
            arr[66][k] = L[2][i];
            i++;
        }
        else {
            arr[0][k] = R[0][j];
            arr[65][k] = R[1][j];
            arr[66][k] = R[2][j];
            j++;
        }
        k++;
    }

    /* Copy the remaining elements of L[], if there
    are any */
    while (i < n1) {
        arr[0][k] = L[0][i];
        arr[65][k] = L[1][i];
        arr[66][k] = L[2][i];
        i++;
        k++;
    }

    /* Copy the remaining elements of R[], if there
    are any */
    while (j < n2) {
        arr[0][k] = R[0][j];
        arr[65][k] = R[1][j];
        arr[66][k] = R[2][j];
        j++;
        k++;
    }
}

#endif // BINARY

/*The following code is for choosing a new base after all distance masks is computed*/


void findBestNewBase(){

    #if RAND==1
    int temp_sum;
    int temp_new_dist[num_of_target];
    int largest_sum = 0;
    int largest_norm_square;
    int number_of_best_pairs = 0;


    //Go through each pair of base, find the largest distance mask and norm
    for(size_t b = 0; b < num_of_new_base;++b){


        temp_sum = 0;
        for(size_t i = 0; i<num_of_target;++i){
            temp_sum += M[i+1][b];
            temp_new_dist[i] = dist_vector[i] - M[i+1][b];
        }


        if(temp_sum > largest_sum){
            largest_norm_square = normSquare(num_of_target,temp_new_dist);
            largest_sum = temp_sum;
            number_of_best_pairs = 1;
        }
        else if(temp_sum == largest_sum && (largest_norm_square < normSquare(num_of_target,temp_new_dist)) ){
            largest_norm_square = normSquare(num_of_target,temp_new_dist);
            largest_sum = temp_sum;
            number_of_best_pairs = 1;
        }
        else if(temp_sum == largest_sum && ( normSquare(num_of_target,temp_new_dist) == largest_norm_square)){
            number_of_best_pairs ++;
        }

    }


    //Put those new bases with the highest sum of distance mask and norm into a set
    int best_pair_index[number_of_best_pairs];
    int temp_count = 0;
    for(size_t b = 0; b < num_of_new_base;++b)
    {

        temp_sum = 0;
        for(size_t i = 0; i<num_of_target;++i){
            temp_sum += M[i+1][b];
            temp_new_dist[i] = dist_vector[i] - M[i+1][b];
        }


        if(temp_sum == largest_sum && ( normSquare(num_of_target,temp_new_dist) == largest_norm_square)){
            best_pair_index[temp_count] = b;
            temp_count++;
        }

    }


    //Randomly select a new base from the set
    srand(clock());
    int choice = rand()%number_of_best_pairs;
    best_new_base_index = best_pair_index[choice];

    #if PRINT_DETAIL
    printf("RAND: %d equally good new bases:\n",number_of_best_pairs);
    for(size_t i = 0; i < number_of_best_pairs; i++)
    {
        printf("%d %d\n", M[65][best_pair_index[i]],M[66][best_pair_index[i]]);
    }
    #endif // PRINT_DETAIL


    #else

    int temp_sum;
    int largest_sum = 0;
    int temp_new_dist[num_of_target];
    int pre_norm;

    //go through each new base, find the the best new base(maximize the distance mask and norm)
    for(size_t b = 0; b < num_of_new_base;++b){


            temp_sum = 0;
            for(size_t i = 0; i<num_of_target;++i){
                temp_sum += M[i+1][b];
                temp_new_dist[i] = dist_vector[i] - M[i+1][b];
            }


            if(temp_sum > largest_sum){
                best_new_base_index = b;
                pre_norm = normSquare(num_of_target,temp_new_dist);
                largest_sum = temp_sum;
            }
            else if(temp_sum == largest_sum && (pre_norm<normSquare(num_of_target,temp_new_dist)) ){
                   best_new_base_index = b;
                    pre_norm = normSquare(num_of_target,temp_new_dist);
                    largest_sum = temp_sum;
            }
    }


    #endif

    //Find the corresponding old bases indexes
    best_pair[0] = M[65][best_new_base_index];
    best_pair[1] = M[66][best_new_base_index];

    //Update dist_vector according to the new base
    for(size_t i = 0; i<num_of_target;++i){
        dist_vector[i] = dist_vector[i] - M[i+1][best_new_base_index];
    }

    //Append the new base
    base_arr[num_of_base] = M[0][best_new_base_index];

}

void findBestNewBaseSorted(){
    #if RAND
        int temp_sum;
        int largest_sum = 0;
        int temp_new_dist[num_of_target];
        int largest_norm_square;
        int number_of_best_pairs = 0;


        //Go through each pair of base, find the largest distance mask and norm
        for(size_t b = 0; b < num_of_new_base;++b){


                temp_sum = 0;
                for(size_t i = 0; i<num_of_target;++i){
                    temp_sum += M_sorted[i+1][b];
                    temp_new_dist[i] = dist_vector[i] - M_sorted[i+1][b];
                }


                if(temp_sum > largest_sum){
                    largest_norm_square = normSquare(num_of_target,temp_new_dist);
                    largest_sum = temp_sum;
                    number_of_best_pairs = 1;
                }
                else if(temp_sum == largest_sum && (largest_norm_square < normSquare(num_of_target,temp_new_dist)) ){
                    largest_norm_square = normSquare(num_of_target,temp_new_dist);
                    largest_sum = temp_sum;
                    number_of_best_pairs = 1;
                }
                else if(temp_sum == largest_sum && ( normSquare(num_of_target,temp_new_dist) == largest_norm_square)){
                    number_of_best_pairs ++;
                }

        }

        //Put those new bases with the highest sum of distance mask and norm into a set
        int best_pair_index[number_of_best_pairs];
        int temp_count = 0;
        for(size_t b = 0; b < num_of_new_base;++b)
        {

                temp_sum = 0;
                for(size_t i = 0; i<num_of_target;++i){
                    temp_sum += M_sorted[i+1][b];
                    temp_new_dist[i] = dist_vector[i] - M_sorted[i+1][b];
                }


                if(temp_sum == largest_sum && ( normSquare(num_of_target,temp_new_dist) == largest_norm_square)){
                    best_pair_index[temp_count] = b;
                    temp_count++;
                }


        }

        //Randomly select a new base from the set
        srand(clock());
        int choice = rand()%number_of_best_pairs;
        best_new_base_index = best_pair_index[choice];

        #if PRINT_DETAIL
        printf("RAND: %d equally good new bases:\n",number_of_best_pairs);
        for(size_t i = 0; i < number_of_best_pairs; i++)
        {
            printf("%d %d\n", M_sorted[65][best_pair_index[i]],M_sorted[66][best_pair_index[i]]);
        }
        #endif // PRINT_DETAIL

    #else
    int temp_sum;
    int largest_sum = 0;
    int temp_new_dist[num_of_target];
    int pre_norm;

    //Go through each new base, find the the best new base
    for(size_t b = 0; b < num_of_new_base;++b){
        temp_sum = 0;
        for(size_t i = 0; i<num_of_target;++i){
            temp_sum += M_sorted[i+1][b];
            temp_new_dist[i] = dist_vector[i] - M_sorted[i+1][b];
        }


        if(temp_sum > largest_sum){
            best_new_base_index = b;
            pre_norm = normSquare(num_of_target,temp_new_dist);
            largest_sum = temp_sum;
        }
        else if(temp_sum == largest_sum && (pre_norm<normSquare(num_of_target,temp_new_dist)) ){
               best_new_base_index = b;
                pre_norm = normSquare(num_of_target,temp_new_dist);
                largest_sum = temp_sum;
        }
    }



    #endif // RAND

    //Find the corresponding old bases indexes
    best_pair[0] = M_sorted[65][best_new_base_index];
    best_pair[1] = M_sorted[66][best_new_base_index];

    //Update dist_vector according to the new base
    for(size_t i = 0; i<num_of_target;++i){
        dist_vector[i] = dist_vector[i] - M_sorted[i+1][best_new_base_index];
    }

    //Append the new base
    base_arr[num_of_base] = M_sorted[0][best_new_base_index];


}

double normSquare(int num_of_target, int dist_vector[]){

    //Return the square of the euclidean norm

    double sum = 0;

    for(size_t i = 0; i < num_of_target;++i)
    {
        sum += dist_vector[i] * dist_vector[i];
    }

    return sum;

}

/*Other useful funcitons*/

int initializeMatrix(int rows, int cols, int(*a)[cols],int value){//Initialize the matrix to "value"

    for(size_t r=0; r < rows; ++r)
    {
        for(size_t c=0; c < cols; ++c){

            a[r][c]=value;
        }
    }
    return 0;
}

int initializeMatrixPartial(int cols, uint32_t (*a)[cols], int prows, int pcols, int value){//Initialize the first "prows" rows and "pcols" cols of the matrix to "value"

    for(size_t r=0; r < prows; ++r)
    {
        for(size_t c=0; c < pcols; ++c){

            a[r][c]=value;
        }
    }
    return 0;
}

int distanceSum(){

    int sum = 0;

    for(size_t i = 0; i < num_of_target; ++i)
    {
        sum += dist_vector[i];
    }


    return sum;
}

int check(){ //Check if the output circuit is correct
//check if the circuit is correct
//return 1 if the circuit is circuit is correct, 0 if incorrect

  int m, n, p, q, c, d, k, sum = 0;
  int first[64][64]={0}, second[64][1]={0}, multiply[64][1]={0};

    m = row;
    n = col;
    p = row;
    q = 1;

  for (c = 0; c < m; c++){
    for (d = 0; d < n; d++){
        first[c][d] = target_matrix[c][d];
    }
  }



  if (m != p)
    printf("The multiplication isn't possible.\n");
  else
  {
    //generate a random p by 1 matrix

    for(size_t i = 0; i < row;i++) {

            srand(time(NULL)+i);
            second[i][0] = rand()%2;
            circuit[i][3] = second[i][0];
    }


    //Calculate the result from original matrix
    for (c = 0; c < m; c++) {
      for (d = 0; d < q; d++) {
        for (k = 0; k < p; k++) {
          sum = (sum + first[c][k]*second[k][d])%2;
        }

        multiply[c][d] = sum;
        sum = 0;
      }
    }
  }

    //Calculate the result from resulted circuit, save to the third col of circuit[][]
    for(size_t i = row; i < num_of_base; i++)
    {
        circuit[i][3] = (circuit[circuit[i][0]][3] + circuit[circuit[i][1]][3])%2;
    }

    //Cross check the results
    for(size_t target_index = 0; target_index < num_of_target; target_index++)
    {
        for(size_t i = row; i < num_of_base; i++)
        {
            if(circuit[i][2] == target_index){
                if(circuit[i][3] != multiply[target_index][0]) {
                    return 0;
                }
            }
        }
    }


  return 1;
}

void stopWatch(int flag){ //0 start, 1 stop, -1 reset, 10 print result

    //Setup timer
    static clock_t start_time, end_time, time_used_clock;
    static double time_used, total_time;


    //Start timer
    if(flag == 0) start_time = clock();

    //Stop timer
    if(flag == 1)
    {
        end_time = clock();
        time_used_clock += end_time - start_time;

    }

    //Reset timer
    if(flag == -1) time_used_clock = 0;

    //Print total time
    if(flag == 10)
    {
        time_used = ((double)time_used_clock) / CLOCKS_PER_SEC;
        printf("Stop watch 1: %fs\n", time_used);
    }
}

void printUint32Arr(const int num_of_element, const uint32_t(*arr)){


    unsigned long long int temp = 0;
    uint32_t reversed_num = 0;

    for(size_t r = 0; r < num_of_element; ++r)
    {
        temp = arr[r];
        reversed_num = 0;

        printf("%3d  ", r);
        //reverse the number so can print from high digit to low digit
        for(size_t c = 0; c < PRINT_DIGITALS; ++c )
        {
            reversed_num = temp%2 + reversed_num * 2;
            temp = temp/2;
        }

        for(size_t i = 0; i< PRINT_DIGITALS; ++i)
        {
            int a = reversed_num%2;
            printf("%d", a);
            reversed_num = reversed_num/2;
        }
        printf("\n");
    }

    return;
}

int printMatrix(size_t rows, size_t cols, int *mat_pointer){

    for(size_t i = 0; i < rows; ++i)
    {
        for(size_t j = 0; j < cols; ++j)
        {
            printf("%d ", *(mat_pointer + (i * cols + j)));
        }
        puts("");
    }
    return 0;
}

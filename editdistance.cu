#include <stdio.h>
#include <time.h>
#include <stdint.h>

#define WINDOW_SIZE 16384

#define NUM_THREADS 128

#define CHUNK_SIZE_A WINDOW_SIZE / NUM_THREADS

#define CHUNK_SIZE_B 64

#define WINDOW_SIZE_B 4096

#define MAX_CORRESPONDING_WINDOWS_B 8

#define MAX_CORRESPONDING_CHUNKS_B 4

#define MOD_MAX_CORRESPONDING_CHUNKS_B 3

#define MAX_LENGTH 400000000

#define CHARS_PER_INT 16

#define CHAR_BITS 4

#define MAX_NUM_CHARS 16


char input_initial_1[MAX_LENGTH], input_initial_2[MAX_LENGTH];

uint64_t newinput_1[MAX_LENGTH / 64 * MAX_NUM_CHARS], newinput_2[MAX_LENGTH / CHARS_PER_INT];

int final_computation_1[MAX_LENGTH /  WINDOW_SIZE_B], final_computation_2[MAX_LENGTH / WINDOW_SIZE_B];

__constant__ uint64_t MAX_INT, HIGH_BIT;
__constant__ int NUM_CHARS;


#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
  #define printf(f, ...) ((void)(f, __VA_ARGS__),0)
#endif


/////////////////////////////////////

// string alignment for two blocks
__device__ void myers_alignment(int length, int la, uint64_t *string_b, uint64_t *vn, uint64_t *vp, uint64_t *hn, uint64_t *hp, uint64_t *d0, int &score, uint64_t *same) {

  for ( int i = 0 ; i < length ; i ++ ) {
    uint64_t ch = (string_b[(i >> 4)] >> ((i & 15) << 2)) & 15;


    uint64_t carry = 0, new_carry;
    uint64_t hp_shift = 1, hn_shift = 0;

    uint64_t *b = same;
    for ( int j = 0 ; j < la ; j ++ ) {
      uint64_t x = (b[ch] | vn[j]);
      uint64_t temp = x & vp[j];

      if ( temp > MAX_INT - carry || temp + carry > MAX_INT - vp[j] ) new_carry = 1; else new_carry = 0;
      temp += (carry + vp[j]);
      carry = new_carry;

      d0[j] = ((temp ^ vp[j])| x);
      hn[j] = vp[j] & d0[j];
      hp[j] = vn[j] | (~(vp[j] | d0[j]));

      uint64_t new_x = ((hp[j] << 1) | hp_shift) ;
      vn[j] = (new_x & d0[j]);
      vp[j] = (((hn[j] << 1) | hn_shift) | (~(new_x | d0[j])));

      hp_shift = ((hp[j] & HIGH_BIT) != 0);
      hn_shift = ((hn[j] & HIGH_BIT) != 0);

      b += NUM_CHARS;
    }

    if (hp_shift) score ++;
    else {
      if (hn_shift) score --;
    }

    
  }
}


__device__ int editdistance_gpu(char* str1, char* str2, int l1, int l2) {
    int* a = (int*)malloc((l2 + 1) * sizeof(int));
    int* b = (int*)malloc((l2 + 1) * sizeof(int));
    int* temp;

    for (int j = 0 ; j <= l2 ; j ++ ) a[j] = j;

    for (int i = 1; i <= l1; i++) {
        b[0] = i;
        for (int j = 1; j <= l2; j++) {
            b[j] = b[j - 1] + 1;
            if ( b[j] > a[j] + 1 ) b[j] = a[j] + 1;
            if (str1[i - 1] == str2[j - 1] && b[j] > a[j-1]) b[j] = a[j - 1];
            if (str1[i - 1] != str2[j - 1] && b[j] > a[j-1] + 1) b[j] = a[j - 1] + 1;

        }
        temp = a;
        a = b;
        b = temp;
    }
    return a[l2];
}



// compute edit distance for two strings
__global__ void new_edit_distance(int l2, uint64_t *input1,  uint64_t *input2,  uint64_t *solution, char *str_1, char *str_2) //, int work_length, int step_size, int shift, int NUM_POSSIBLE_END_POINTS)
{ 

   __shared__ int table[MAX_CORRESPONDING_WINDOWS_B][NUM_THREADS + 1][MAX_CORRESPONDING_CHUNKS_B];
   int temp_table[MAX_CORRESPONDING_WINDOWS_B];

   for ( int i = 0 ; i < MAX_CORRESPONDING_WINDOWS_B ; i ++ ) for ( int k = 0 ; k < MAX_CORRESPONDING_CHUNKS_B ; k ++ ) {
     table[i][threadIdx.x + 1][k] = 491510000;
     if ( threadIdx.x == 0 ) table[i][0][k] = 491510000;
   }
   table[0][0][MAX_CORRESPONDING_CHUNKS_B - 1] = 0;
   table[0][threadIdx.x + 1][ MAX_CORRESPONDING_CHUNKS_B - 1] = (threadIdx.x + 1) * CHUNK_SIZE_A;

   __syncthreads(); 

   int temp_shift_1 = (blockIdx.x * WINDOW_SIZE + threadIdx.x * CHUNK_SIZE_A) / 64;


   uint64_t vn[MAX_CORRESPONDING_CHUNKS_B][CHUNK_SIZE_A / 64], vp[MAX_CORRESPONDING_CHUNKS_B][CHUNK_SIZE_A / 64], hp[MAX_CORRESPONDING_CHUNKS_B][CHUNK_SIZE_A / 64], 
            hn[MAX_CORRESPONDING_CHUNKS_B][CHUNK_SIZE_A / 64], d0[MAX_CORRESPONDING_CHUNKS_B][CHUNK_SIZE_A / 64];
   int score[MAX_CORRESPONDING_CHUNKS_B];

   for ( int i = 0 ; i < MAX_CORRESPONDING_CHUNKS_B ; i ++ )  {
     for ( int j = 0 ; j < CHUNK_SIZE_A / 64 ; j ++ ) {
       vn[i][j] = 0;
       vp[i][j] = MAX_INT;
       hp[i][j] = 0;
       hn[i][j] = 0;
       d0[i][j] = 0;
     }
     score[i] =  491510000;
   }
   score[0] = CHUNK_SIZE_A;


   for (  int i = 0 ; i < l2 / CHUNK_SIZE_B; i ++ ) {

     for ( int j = 0 ; j < MAX_CORRESPONDING_CHUNKS_B ; j ++ ) {
       myers_alignment(CHUNK_SIZE_B, CHUNK_SIZE_A / 64, input2 + i * CHUNK_SIZE_B / CHARS_PER_INT, vn[j], vp[j], hn[j], hp[j], d0[j], score[j], input1 + temp_shift_1 * NUM_CHARS);
     }

     __syncthreads();



     for ( int j = 0 ; j < MAX_CORRESPONDING_WINDOWS_B ; j ++ ) { 
       temp_table[j] = table[j][threadIdx.x + 1][(i + MAX_CORRESPONDING_CHUNKS_B - 1) & MOD_MAX_CORRESPONDING_CHUNKS_B] + CHUNK_SIZE_B; // current chunk of B corresponds to empty in A
       for ( int k = 0 ; k < MAX_CORRESPONDING_CHUNKS_B ; k ++ ) {
         if ( temp_table[j] > table[j][threadIdx.x][k] + score[(k + 1) & MOD_MAX_CORRESPONDING_CHUNKS_B] ) 
           temp_table[j] = table[j][threadIdx.x][k] + score[(k + 1) & MOD_MAX_CORRESPONDING_CHUNKS_B];
       }
     }
     
     __syncthreads();
     


     for ( int j = 0 ; j < MAX_CORRESPONDING_WINDOWS_B ; j ++ ) {
       if ( threadIdx.x == 0 ) table[j][0][i & MOD_MAX_CORRESPONDING_CHUNKS_B] = table[j][0][(i + MAX_CORRESPONDING_CHUNKS_B - 1) & MOD_MAX_CORRESPONDING_CHUNKS_B] + CHUNK_SIZE_B;
       table[j][threadIdx.x + 1][i & MOD_MAX_CORRESPONDING_CHUNKS_B] = temp_table[j];
     }
     __syncthreads();

     for ( int j = 0 ; j < MAX_CORRESPONDING_WINDOWS_B ; j ++ ) {
       for ( int k = threadIdx.x ; k >= 0 && k >= threadIdx.x - 20 ; k -- ) { 
         if ( table[j][threadIdx.x + 1][i & MOD_MAX_CORRESPONDING_CHUNKS_B] > table[j][k][i & MOD_MAX_CORRESPONDING_CHUNKS_B] + (threadIdx.x + 1 - k) * CHUNK_SIZE_A ) 
           table[j][threadIdx.x + 1][i & MOD_MAX_CORRESPONDING_CHUNKS_B] = table[j][k][i & MOD_MAX_CORRESPONDING_CHUNKS_B] + (threadIdx.x + 1 - k) * CHUNK_SIZE_A;
       }
     }

     for ( int j = 0 ; j < CHUNK_SIZE_A / 64 ; j ++ ) {
       vn[(i + 1) % MAX_CORRESPONDING_CHUNKS_B][j] = 0;
       vp[(i + 1) % MAX_CORRESPONDING_CHUNKS_B][j] = MAX_INT;
       hp[(i + 1) % MAX_CORRESPONDING_CHUNKS_B][j] = 0;
       hn[(i + 1) % MAX_CORRESPONDING_CHUNKS_B][j] = 0;
       d0[(i + 1) % MAX_CORRESPONDING_CHUNKS_B][j] = 0;
     }
     score[(i + 1) % MAX_CORRESPONDING_CHUNKS_B] = CHUNK_SIZE_A;

     __syncthreads();

     if ( i % (WINDOW_SIZE_B / CHUNK_SIZE_B) == WINDOW_SIZE_B / CHUNK_SIZE_B - 1 ) { 
       if ( threadIdx.x == NUM_THREADS - 1 ) { 
         int v = i / (WINDOW_SIZE_B / CHUNK_SIZE_B);
         for ( int k = 0 ; k < MAX_CORRESPONDING_WINDOWS_B ; k ++ ) { 
           if ( v >= 0 ) {
             solution[blockIdx.x * (l2 / WINDOW_SIZE_B) * MAX_CORRESPONDING_WINDOWS_B + v * MAX_CORRESPONDING_WINDOWS_B + i / (WINDOW_SIZE_B / CHUNK_SIZE_B) - v] 
               = table[v % MAX_CORRESPONDING_WINDOWS_B][NUM_THREADS][i % MAX_CORRESPONDING_CHUNKS_B]; 
           }
           v--;
         }  
       }

       int v = (i / (WINDOW_SIZE_B / CHUNK_SIZE_B) + 1) % MAX_CORRESPONDING_WINDOWS_B;
       for ( int k = 0 ; k < MAX_CORRESPONDING_CHUNKS_B ; k ++ ) {
         table[v][threadIdx.x + 1][k] = 491510000;
         if ( threadIdx.x == 0 ) table[v][0][k] = 491510000;
       }
       table[v][0][MAX_CORRESPONDING_CHUNKS_B - 1] = 0;
       table[v][threadIdx.x + 1][ MAX_CORRESPONDING_CHUNKS_B - 1] = (threadIdx.x + 1) * CHUNK_SIZE_A;

       __syncthreads();
     }

  }
}



int editdistance_cpu(char* str1, char* str2, int l1, int l2) {
    int* a = (int*)malloc((l2 + 1) * sizeof(int));
    int* b = (int*)malloc((l2 + 1) * sizeof(int));
    int* temp;

    for (int j = 0 ; j <= l2 ; j ++ ) a[j] = j;

    for (int i = 1; i <= l1; i++) {
if ( i % 1000 == 0 ) printf("%d %d\n", i, l1);
        b[0] = i;
        for (int j = 1; j <= l2; j++) {
            b[j] = b[j - 1] + 1;
            if ( b[j] > a[j] + 1 ) b[j] = a[j] + 1;
            if (str1[i - 1] == str2[j - 1] && b[j] > a[j-1]) b[j] = a[j - 1];
            if (str1[i - 1] != str2[j - 1] && b[j] > a[j-1] + 1) b[j] = a[j - 1] + 1;

        }
        temp = a;
        a = b;
        b = temp;
    }
    return a[l2];
}


int main(void)
{
  // read file
   int cc = 0;
  freopen("file1.fasta", "r", stdin);
  char header1[500];
  fgets(header1, 500, stdin);
  while (scanf("%s", input_initial_1+cc) != EOF) {
     int t = strlen(input_initial_1+cc);
    cc += t;
  }

  freopen("file2.fasta", "r", stdin);
  char header2[500];
  fgets(header2, 500, stdin);
  cc = 0;
  while (scanf("%s", input_initial_2+cc) != EOF) {
     int t = strlen(input_initial_2+cc);
    cc += t;
  }

  int l1 = strlen(input_initial_1);
  int l2 = strlen(input_initial_2);

  printf("string 1 length = %d      string 2 length = %d\n", l1, l2);

  //count characters

   int used_chars[300];
   int char_count = 0;
  for (  int i = 0 ; i < 300; i ++ ) used_chars[i] = -1;
  for (  int i = 0 ; i < l1 ; i ++ ) 
    if ( used_chars[input_initial_1[i]] < 0 ) {
      char_count ++;
      used_chars[input_initial_1[i]] = char_count - 1;
    }
  
  for (  int i = 0 ; i < l2 ; i ++ ) 
    if ( used_chars[input_initial_2[i]] < 0 ) {
      char_count ++;
      used_chars[input_initial_2[i]] = char_count - 1;
    }
  if (char_count > (1 << CHAR_BITS) ) {
    printf("too many different characters!\n");
    return 0;
  }
  

  // convert input to int

  for ( int i = 0 ; i < l1 / 64 ; i ++ ) {
    for ( int j = 0 ; j < char_count ; j ++ ) {
      uint64_t s = 0;
      for ( int k = (i+1) * 64 - 1 ; k >= i * 64 ; k -- ) {
        s <<= 1;
        if ( used_chars[input_initial_1[k]] == j ) s += 1;
      }
      newinput_1[i * char_count + j] = s;
    }
  }
  for ( int i = 0 ; i < l2 / CHARS_PER_INT ; i ++ ) {
    newinput_2[i] = 0;
    for (  int j = 0 ; j < CHARS_PER_INT ; j ++ ) {
      newinput_2[i] |= (((uint64_t)used_chars[input_initial_2[i * CHARS_PER_INT + j]]) << (j * 4));
    }
  }


  //copy input to CUDA memory
  uint64_t *str1, *str2;

  cudaMalloc(&str1, MAX_LENGTH / 64 * MAX_NUM_CHARS *sizeof(uint64_t)); 
  cudaMalloc(&str2, MAX_LENGTH / CHARS_PER_INT *sizeof(uint64_t));
  cudaMemcpy(str1, newinput_1, MAX_LENGTH / 64 * MAX_NUM_CHARS *sizeof(uint64_t), cudaMemcpyHostToDevice);
  cudaMemcpy(str2, newinput_2, MAX_LENGTH / CHARS_PER_INT*sizeof(uint64_t), cudaMemcpyHostToDevice);




  char *real_str1, *real_str2;
  cudaMalloc(&real_str1, MAX_LENGTH * sizeof(char));
  cudaMalloc(&real_str2, MAX_LENGTH * sizeof(char));
  cudaMemcpy(real_str1, input_initial_1, MAX_LENGTH * sizeof(char), cudaMemcpyHostToDevice);
  cudaMemcpy(real_str2, input_initial_2, MAX_LENGTH * sizeof(char), cudaMemcpyHostToDevice);





  const uint64_t max_int = UINT64_MAX;
  const uint64_t high_bit = ((uint64_t)1) << 63;
  cudaMemcpyToSymbol(MAX_INT, &max_int, sizeof(uint64_t));
  cudaMemcpyToSymbol(HIGH_BIT, &high_bit, sizeof(uint64_t));
  cudaMemcpyToSymbol(NUM_CHARS, &char_count, sizeof(int));

  int solution_size = 400000000;
  printf("GPU solution size = %d\n", solution_size);

  uint64_t *final_solution, *solution;
  cudaMalloc(&solution, solution_size*sizeof(uint64_t));
  final_solution = (uint64_t*)malloc(solution_size*sizeof(uint64_t));
 
  printf("%d   %d    %d                  %d   %d   %d\n", WINDOW_SIZE, l1, l1 / WINDOW_SIZE, WINDOW_SIZE_B, l2, l2 / WINDOW_SIZE_B);

  printf("start parallel\n");


  
  new_edit_distance<<<l1 / WINDOW_SIZE, NUM_THREADS>>>(l2, str1, str2, solution, real_str1, real_str2); 
 


  cudaError_t errSync  = cudaGetLastError();
  cudaError_t errAsync = cudaDeviceSynchronize();
  if (errSync != cudaSuccess) 
    printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
  if (errAsync != cudaSuccess)
    printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));

  cudaMemcpy(final_solution, solution, solution_size*sizeof( int), cudaMemcpyDeviceToHost);

  printf("end parallel\n");

  // final solution computation

  int *s1, *s2, *temp_pointer; 
  s1 = final_computation_1;
  s2 = final_computation_2;
  s1[0] = 0;
  for ( int i = 1 ; i <= l2 / WINDOW_SIZE_B ; i ++ ) s1[i] = WINDOW_SIZE_B * i;

  for ( int i = 0 ; i < l1 / WINDOW_SIZE ; i ++ ) { 

     s2[0] = (i + 1) * WINDOW_SIZE;
     for ( int j = 0 ; j < l2 / WINDOW_SIZE_B ; j ++ ) {
       s2[j + 1] = s1[j + 1] + WINDOW_SIZE; // current window of A corresponds to empy in B

       for ( int k = 0 ; k < MAX_CORRESPONDING_WINDOWS_B ; k ++ ) {
         // current window of A corresponds to something non-trivial in B

         if ( j - k >= 0 && s2[j + 1] > s1[j - k] + final_solution[i * (l2 / WINDOW_SIZE_B) * MAX_CORRESPONDING_WINDOWS_B + (j - k) * MAX_CORRESPONDING_WINDOWS_B + k] )
           s2[j + 1] = s1[j - k] + final_solution[i * (l2 / WINDOW_SIZE_B) * MAX_CORRESPONDING_WINDOWS_B + (j - k) * MAX_CORRESPONDING_WINDOWS_B + k] ;


       }
     }
     temp_pointer = s1; s1 = s2; s2 = temp_pointer;     

  }
  printf("final edit distance = %d\n", s1[l2 / WINDOW_SIZE_B]);

  cudaDeviceReset();
  cudaFree(str1);
  cudaFree(str2);
  cudaFree(solution);
  free(final_solution);
}


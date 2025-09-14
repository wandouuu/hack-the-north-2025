/*
 * camera streamer
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <netdb.h>
#include <termios.h>
#include <time.h>
#include <camera/camera_api.h>

// default config
#define DEFAULT_SERVER_PORT 12345
#define DEFAULT_DOWNSCALE_FACTOR 12

static int g_sock = -1;
static uint8_t* g_small_frame_buffer = NULL;

// parameters
static char g_server_ip[256] = "10.37.116.174";
static int g_target_fps = 10;
static int g_downscale_width = 192;
static int g_downscale_height = 108;

// FPS timing variables
static struct timespec g_last_frame_time = {0, 0};
static long g_frame_interval_ns = 100000000L; // Will be calculated based on FPS

/**
 * @brief Get current time in nanoseconds
 */
static long long get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

/**
 * @brief Check if enough time has passed since last frame
 */
static int should_send_frame(void) {
    struct timespec current_time;
    clock_gettime(CLOCK_MONOTONIC, &current_time);

    long long current_ns = (long long)current_time.tv_sec * 1000000000LL + current_time.tv_nsec;
    long long last_ns = (long long)g_last_frame_time.tv_sec * 1000000000LL + g_last_frame_time.tv_nsec;

    if (current_ns - last_ns >= g_frame_interval_ns) {
        g_last_frame_time = current_time;
        return 1;
    }
    return 0;
}

/**
 * @brief Send frame with header for synchronization
 */
static int send_frame_with_header(uint8_t* frame_data, uint32_t frame_size) {
    // Send a simple header: magic number + frame size
    uint32_t magic = 0xDEADBEEF;

    if (write(g_sock, &magic, sizeof(magic)) != sizeof(magic)) {
        return -1;
    }
    if (write(g_sock, &frame_size, sizeof(frame_size)) != sizeof(frame_size)) {
        return -1;
    }
    if (write(g_sock, frame_data, frame_size) != frame_size) {
        return -1;
    }

    return 0;
}

void downscale_and_pack_frame(const camera_buffer_t* big_frame, uint8_t* small_buffer) {
    uint32_t src_width = big_frame->framedesc.bgr8888.width;
    uint32_t src_height = big_frame->framedesc.bgr8888.height;
    const uint8_t* src_buf = (const uint8_t*)big_frame->framebuf;
    
    // downscale factors dynamically
    int scale_factor_x = src_width / g_downscale_width;
    int scale_factor_y = src_height / g_downscale_height;
    
    if (scale_factor_x < 1) scale_factor_x = 1;
    if (scale_factor_y < 1) scale_factor_y = 1;
    
    int small_buf_idx = 0;

    for (int y = 0; y < g_downscale_height; ++y) {
        for (int x = 0; x < g_downscale_width; ++x) {
            // Calculate source pixel position using dynamic scale factors
            int src_y = y * scale_factor_y;
            int src_x = x * scale_factor_x;
            
            if (src_y >= src_height) src_y = src_height - 1;
            if (src_x >= src_width) src_x = src_width - 1;
            
            const uint8_t* src_pixel_ptr = &src_buf[src_y * src_width * 4 + src_x * 4];

            small_buffer[small_buf_idx++] = src_pixel_ptr[0]; // Blue
            small_buffer[small_buf_idx++] = src_pixel_ptr[1]; // Green
            small_buffer[small_buf_idx++] = src_pixel_ptr[2]; // Red
        }
    }
}

static void processCameraData(camera_handle_t handle, camera_buffer_t* buffer, void* arg) {
    if (g_sock < 0 || g_small_frame_buffer == NULL) return;

    // only process frame if enough time has passed
    if (!should_send_frame()) {
        return;
    }

    if (buffer->frametype == CAMERA_FRAMETYPE_BGR8888) {
        downscale_and_pack_frame(buffer, g_small_frame_buffer);

        uint32_t small_frame_size = g_downscale_width * g_downscale_height * 3;

        // Send frame with header for sync
        if (send_frame_with_header(g_small_frame_buffer, small_frame_size) < 0) {
            fprintf(stderr, "Failed to send frame\n");
        }
    }
}

void print_usage(const char* program_name) {
    printf("Usage: %s [options]\n", program_name);
    printf("Options:\n");
    printf("  -i <ip_address>    Server IP address (default: %s)\n", g_server_ip);
    printf("  -f <fps>           Target FPS (default: %d)\n", g_target_fps);
    printf("  -w <width>         Frame width (default: %d)\n", g_downscale_width);
    printf("  -h <height>        Frame height (default: %d)\n", g_downscale_height);
    printf("  -?                 Show this help message\n");
    printf("\nExample: %s -i 192.168.1.100 -f 15 -w 320 -h 240\n", program_name);
}

int main(int argc, char* argv[]) {
    camera_handle_t handle = CAMERA_HANDLE_INVALID;
    struct sockaddr_in server;
    struct hostent *hp;
    int opt;

    // Parse CLI args
    while ((opt = getopt(argc, argv, "i:f:w:h:?")) != -1) {
        switch (opt) {
            case 'i':
                strncpy(g_server_ip, optarg, sizeof(g_server_ip) - 1);
                g_server_ip[sizeof(g_server_ip) - 1] = '\0';
                break;
            case 'f':
                g_target_fps = atoi(optarg);
                if (g_target_fps <= 0 || g_target_fps > 60) {
                    fprintf(stderr, "Error: FPS must be between 1 and 60\n");
                    return EXIT_FAILURE;
                }
                break;
            case 'w':
                g_downscale_width = atoi(optarg);
                if (g_downscale_width <= 0) {
                    fprintf(stderr, "Error: Width must be positive\n");
                    return EXIT_FAILURE;
                }
                break;
            case 'h':
                g_downscale_height = atoi(optarg);
                if (g_downscale_height <= 0) {
                    fprintf(stderr, "Error: Height must be positive\n");
                    return EXIT_FAILURE;
                }
                break;
            case '?':
                print_usage(argv[0]);
                return EXIT_SUCCESS;
            default:
                print_usage(argv[0]);
                return EXIT_FAILURE;
        }
    }

    // frame interval from target FPS
    g_frame_interval_ns = 1000000000L / g_target_fps;

    printf("Camera Streamer Configuration:\n");
    printf("  Server IP: %s\n", g_server_ip);
    printf("  Target FPS: %d\n", g_target_fps);
    printf("  Frame size: %dx%d\n", g_downscale_width, g_downscale_height);
    printf("  Frame interval: %ld ns\n", g_frame_interval_ns);

    g_small_frame_buffer = malloc(g_downscale_width * g_downscale_height * 3);
    if (!g_small_frame_buffer) {
        fprintf(stderr, "Failed to allocate memory.\n");
        return EXIT_FAILURE;
    }

    // --- Setup TCP Socket Connection ---
    printf("Connecting to %s:%d...\n", g_server_ip, DEFAULT_SERVER_PORT);
    g_sock = socket(AF_INET, SOCK_STREAM, 0);
    if (g_sock < 0) {
        perror("ERROR creating socket");
        free(g_small_frame_buffer);
        return EXIT_FAILURE;
    }

    hp = gethostbyname(g_server_ip);
    if (hp == NULL) {
        fprintf(stderr, "ERROR: Could not resolve hostname %s\n", g_server_ip);
        close(g_sock);
        free(g_small_frame_buffer);
        return EXIT_FAILURE;
    }

    server.sin_family = AF_INET;
    memcpy(&server.sin_addr, hp->h_addr, hp->h_length);
    server.sin_port = htons(DEFAULT_SERVER_PORT);
    if (connect(g_sock, (struct sockaddr *)&server, sizeof(server)) < 0) {
        perror("ERROR connecting");
        close(g_sock);
        free(g_small_frame_buffer);
        return EXIT_FAILURE;
    }

    // TCP optimizations for low latency
    int flag = 1;
    setsockopt(g_sock, IPPROTO_TCP, TCP_NODELAY, (char *)&flag, sizeof(int));

    // Set send buffer size to reduce buffering
    int sendbuf = 65536; // 64KB
    setsockopt(g_sock, SOL_SOCKET, SO_SNDBUF, &sendbuf, sizeof(sendbuf));

    printf(" -> Connected.\n");

    // --- Setup Camera ---
    if (camera_open(CAMERA_UNIT_1, CAMERA_MODE_RW, &handle) != CAMERA_EOK) {
        fprintf(stderr, "Failed to open camera.\n");
        close(g_sock);
        free(g_small_frame_buffer);
        return EXIT_FAILURE;
    }
    printf(" -> Camera opened.\n");

    if (camera_start_viewfinder(handle, &processCameraData, NULL, NULL) != CAMERA_EOK) {
        fprintf(stderr, "Failed to start viewfinder.\n");
        camera_close(handle);
        close(g_sock);
        free(g_small_frame_buffer);
        return EXIT_FAILURE;
    }

    printf("\nStreaming at %d FPS... Press any key to stop.\n", g_target_fps);
    struct termios oldterm, newterm;
    tcgetattr(STDIN_FILENO, &oldterm);
    newterm = oldterm;
    newterm.c_lflag &= ~(ECHO | ICANON);
    tcsetattr(STDIN_FILENO, TCSANOW, &newterm);
    read(STDIN_FILENO, NULL, 1);
    tcsetattr(STDIN_FILENO, TCSANOW, &oldterm);

    printf("\nStopping...\n");
    camera_stop_viewfinder(handle);
    camera_close(handle);
    close(g_sock);
    free(g_small_frame_buffer);
    return EXIT_SUCCESS;
}

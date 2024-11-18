/**
 * Tetromino (Tetris 4 block pieces)
 * 
 * @author Jean-Charles Quinton
 * @version 07/06/2011
 * creation 07/06/2011
 */

public class Tetromino extends Layer {
    /** Tetromino blocks */
    int[][][][] blocks = {
    { // vertical orientation
        {{0,1,0,0}, // I
         {0,1,0,0},
         {0,1,0,0},
         {0,1,0,0}},
        {{0,1,0,0}, // J
         {0,1,0,0},
         {1,1,0,0},
         {0,0,0,0}},
        {{0,1,0,0}, // L
         {0,1,0,0},
         {0,1,1,0},
         {0,0,0,0}},
        {{0,0,0,0}, // O
         {0,1,1,0},
         {0,1,1,0},
         {0,0,0,0}},
        {{0,1,0,0}, // S
         {1,1,0,0},
         {1,0,0,0},
         {0,0,0,0}},
        {{0,1,0,0}, // Z
         {0,1,1,0},
         {0,0,1,0},
         {0,0,0,0}},
        {{0,1,0,0}, // T
         {0,1,1,0},
         {0,1,0,0},
         {0,0,0,0}}
    },{ // horizontal orientation
        {{0,0,0,0}, // I
         {1,1,1,1},
         {0,0,0,0},
         {0,0,0,0}},
        {{0,0,0,0}, // J
         {1,1,1,0},
         {0,0,1,0},
         {0,0,0,0}},
        {{0,0,1,0}, // L
         {1,1,1,0},
         {0,0,0,0},
         {0,0,0,0}},
        {{0,0,0,0}, // O
         {0,1,1,0},
         {0,1,1,0},
         {0,0,0,0}},
        {{0,0,0,0}, // S
         {1,1,0,0},
         {0,1,1,0},
         {0,0,0,0}},
        {{0,1,1,0}, // Z
         {1,1,0,0},
         {0,0,0,0},
         {0,0,0,0}},
        {{0,1,0,0}, // T
         {1,1,1,0},
         {0,0,0,0},
         {0,0,0,0}}
    }};
    
    /** Tetromino piece type (from I to T) */
    int type;
    /** Center of the piece */
    int cx;
    int cy;
    /** Orientation (0 to 3) */
    int orientation;

    /** Default constructor for the tetromino */
    public Tetromino(int cx, int cy) {
        // Select a random piece
        this(cx,cy,randomType());
    }
    
    /** Random type */
    public static int randomType() {
        return new java.util.Random().nextInt(7)+I;
    }
    
    /** Constructor of the tetromino with default orientation */
    public Tetromino(int cx, int cy, int type) {
        this(cx,cy,type,0);
    }
    
    /** Constructor of the tetromino with default orientation */
    public Tetromino(int cx, int cy, int type, int orientation) {
        // Correct the parameter values
        if (type<=I || type>T) type = I;
        orientation = orientation%4;
        // Set the parameter values
        this.type = type;
        this.orientation = orientation;
        this.cx = cx;
        this.cy = cy;
    }
    
    /** Constructor from another tetromino */
    public Tetromino(Tetromino t) {
        this(t.cx, t.cy, t.type, t.orientation);
    }
    
    @Override
    public int get(int x, int y) {
        // Index to access the pieces content
        int t = type-I;
        // Relative coordinates
        x = (x-cx);
        y = (y-cy);
        // Symmetries based on orientation
        if (orientation>=2 && type!=O) {
            x=-x;
            y=-y;
        }
        // Coordinates in the integer matrices
        x++;
        y++;
        // Orientation index
        int o = orientation%2;
        // Return the block value
        if (x>=0 && x<4 && y>=0 && y<4 && blocks[o][t][y][x]!=0)
            return type;
        else
            return NONE;
    }
    
    /** Rotate the tetromino by 90° clockwise */
    public void rotate(int da) {
        orientation = (orientation+da+4)%4;
    }
    
    /** Move the tetromino by a given vector */
    public void move(int dx) {
        cx += dx;
    }
    
    /** Move the tetromino by a given vector */
    public void fall() {
        cy--;
    }
    
    /** Get a bounding box around the tetromino */
    public View getBounds() {
        return new View(cx-2,cy-2,5,5);
    }
}
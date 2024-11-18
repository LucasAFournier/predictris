/**
 * Tetris simulation Applet
 * (based on courbe_4.0 and cnft_gen_2.2)
 * 
 * @author Jean-Charles Quinton
 * @version 07/06/2011
 * creation 24/01/2008
 */

import java.awt.*;
import java.awt.event.*;
import java.awt.image.*;
import java.awt.geom.*;
import javax.swing.*;
import java.util.*;
import applet.*;

import static java.lang.Math.*;
import static applet.AppletStub.*;

public class TetrisApplet extends JApplet {
    /** Update period in seconds
     * (negative = controlled by the user) */
    double dt = -1;
    /** How often to refresh the view
     * (number of cycles to perform) */
    int refresh = 1;
    
    /** Radius of the agent's view */
    int view_radius = 1;
    
    /** Tetris model */
    Tetris tetris;
    /** Anticipation model */
    Predictor predictor;
        
    /** Update thread */
    java.util.Timer updater;
    
    /** Swing view of the full Tetris model */
    ViewSwing fview;
    
    /** Perception view of the user */
    ViewSwing pview;
    
    /** Actions to perform */
    public static final int NONE = 0;
    public static final int TL = 1; // move tetromino
    public static final int TR = 2;
    public static final int TRL = 3; // rotate tetromino
    public static final int TRR = 4;
    public static final int VL = 5; // move the view
    public static final int VR = 6;
    public static final int VU = 7;
    public static final int VD = 8;

    @Override
    public void init() {
        // Read the parameters
        String s = getParameter("dt");
        if (s!=null)
            dt = valueToDouble(s);
        s = getParameter("refresh");
        if (s!=null && dt>=0)
            refresh = (int)(double)valueToDouble(s);
        // Initialize the Tetris model
        tetris = new Tetris(10,20);
        // Initialize a prediction model
        predictor = new Predictor((int)Math.pow(2*view_radius+1,2),VD);
        //Execute a job on the event-dispatching thread:
        //creating this applet's GUI.
        try {
            javax.swing.SwingUtilities.invokeAndWait(new Runnable() {
                public void run() {
                    // Initialize the GUI                    
                    createGUI();
                }
            });
        } catch (Exception e) {
            System.err.println("The initialization did not successfully complete");
            e.printStackTrace();
        }
    }

    /** Constructor */
    public void createGUI() {
        // Change the background color for all panels
        // (better when displayed as an Applet on my white webpage)
        UIManager.put("Panel.background", Color.white);

        int view_size = 2*view_radius+1;
        View view = new View(-view_radius,20-1-view_size,view_size,view_size);
        view.get(tetris);
        pview = new ViewSwing(view);
        pview.setPreferredSize(new Dimension(400,400));
        getContentPane().add(pview,BorderLayout.CENTER);

        view = new View(-6,0,14,20);
        view.get(tetris);
        fview = new ViewSwing(view);
        fview.subview = pview.view;
        fview.setPreferredSize(new Dimension(300,400));
        getContentPane().add(fview,BorderLayout.WEST);

//         sview.addMouseListener(new MouseAdapter() {
//             @Override
//             public void mouseClicked(MouseEvent e) {
//                 if (updater==null) {
//                     // Update thread initialization and start
//                     updater = new java.util.Timer();
//                     long ms = (long)(dt*1000);
//                     if (ms<0) ms = 10;
//                     updater.scheduleAtFixedRate(new TimerTask() {
//                         public void run() {
// //                             model.update(dt);
// //                             model.updateTraces();
//                             System.out.println("run");
//                             repaint();
//                         }
//                     },0,ms);     
//                 } else {
//                     updater.cancel();
//                     updater=null;
//                 }
//             }
//         });

        addKeyListener(new KeyAdapter() {
            @Override
            public void keyPressed(KeyEvent evt) {
                // Save previous state
                View old_view = new View(pview.view);
                old_view.get(tetris);
                // Variable to determine the action code
                int action = NONE;
                switch(evt.getKeyCode()) {
                    case KeyEvent.VK_LEFT:  tetris.moveTetromino(-1);   action=TL; break;
                    case KeyEvent.VK_RIGHT: tetris.moveTetromino(1);    action=TR; break;
                    case KeyEvent.VK_UP:    tetris.rotateTetromino(1);  action=TRR; break;
                    case KeyEvent.VK_DOWN:  tetris.rotateTetromino(-1); action=TRL; break;
                    case KeyEvent.VK_SPACE: tetris.fallTetromino();     break;
                    case KeyEvent.VK_Q:     moveView(-1,0);             action=VL; break;
                    case KeyEvent.VK_D:     moveView(1,0);              action=VR; break;
                    case KeyEvent.VK_Z:     moveView(0,1);              action=VU; break;
                    case KeyEvent.VK_S:     moveView(0,-1);             action=VD; break;
                    case KeyEvent.VK_P:     tetris.tetromino.type = Tetromino.randomType(); break;
                    case KeyEvent.VK_M:     predictor.display(); break;
                }
                // Update the views
                pview.view.get(tetris);
                fview.view.get(tetris);
                // Update the prediction model
                if (action!=NONE) {
                    predictor.update(old_view.blocks,action,pview.view.blocks);
                }
                // Repaint the components
                repaint();
            }
        });
    }
    
    /** Move the perceptive view from the agent */
    public void moveView(int dx, int dy) {
        pview.view.translate(dx,dy);
    }
    
    @Override
    public void destroy() {
        if (updater!=null) {
            updater.cancel();
        }
    }

    /** Main function to start the program */
    public static void main(String args[]){
        final TetrisApplet applet = new TetrisApplet();
        AppletStub.argsToProperties(args);
        // Configure the frame to display the Applet
        applet.setStub(new AppletStub(applet, "Tetris Applet"));
    }
}
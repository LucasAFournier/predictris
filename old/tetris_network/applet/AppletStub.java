/**
 * (c) Copyright 1999, 2003 Uwe Voigt
 * All Rights Reserved.
 * 
 * Modifications
 * @author Jean-Charles Quinton
 * @version 03/03/10
 */
package applet;

import java.applet.Applet;
import javax.swing.*;
import java.awt.event.*;
import java.awt.*;
import java.net.*;
import java.io.*;

public final class AppletStub extends JFrame implements java.applet.AppletStub, Runnable {
    private Applet applet;
    private URL docBase;
    private java.applet.AppletContext context;
    private String title;

    public AppletStub(final Applet applet, String title) {
        super(title);
        this.title = title;
        this.applet = applet;
        try {
            File file = new File(".");
            // DEPRECATED: docBase = new URL("file", "/", file.getAbsolutePath().replace('\\', '/'));
            // (see https://stackoverflow.com/questions/6098472/pass-a-local-file-in-to-url-in-java)
            docBase = file.toURI().toURL(); // from Java 11
        } catch (Exception e) {
            e.printStackTrace();
        }
        context = new AppletContext(this);
                    
        new Thread(this, "AppletStub").start();
    }
    
    /** Convert command line arguments into system properties */
    public static void argsToProperties(String[] args) {
        // Go through all the parameters
        for (int i=0; i<args.length; i++) {
            String[] nameval = args[i].split("=");
            // Does it have the parameter format : name=value
            if (nameval.length==2) {
                // Remove the '," and additional spaces if needed
                nameval[1].replace('"',' ');
                nameval[1].replace('\'',' ');
                // Set the property
                System.setProperty(nameval[0].trim(),nameval[1].trim());
            }
        }
    }
    
    /** Convert a parameter value into a boolean */
    public static boolean valueToBoolean(String val) {
        if (val==null) return false;
        return val.equals("true") || val.equals("1") || val.equals("yes");
    }
    
    /** Convert a parameter value into a double */
    public static Double valueToDouble(String val) {
        Double d = null;
        try {
            d = Double.parseDouble(val);
        } catch (Exception e) {
            // This includes the case where "val" is null
        }
        return d;
    }

    @Override
    public void run() {
        applet.init();
        add(applet);
        
        addWindowListener (new WindowAdapter() {
            public void windowClosing (WindowEvent event) {
                applet.stop();
                applet.destroy();
                dispose();
                System.exit(0);
            }
        });
        
        pack();
        setVisible(true);
        applet.start();
    }

    @Override
    public void appletResize(int w, int h) {
        applet.resize(w, h);
    }

    @Override
    public java.applet.AppletContext getAppletContext() {
        return context;
    }

    @Override
    public URL getCodeBase() {
        return docBase;
    }

    @Override
    public URL getDocumentBase() {
        return docBase;
    }

    @Override
    public String getParameter(String name) {
        return System.getProperty(name);
    }

    @Override
    public boolean isActive() {
        return true;
    }

    @Override
    public void setSize(int w, int h) {
        super.setSize(w, h);
        Insets i = getInsets();
        appletResize(w - i.left - i.right, h - i.top - i.bottom);
    }
}

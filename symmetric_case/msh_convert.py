import subprocess, os
from dolfin import Mesh, HDF5File, MeshFunction


def convert(msh_file, h5_file):
    '''Convert msh file to h5_file'''
    root, _ = os.path.splitext(msh_file)
    assert os.path.splitext(msh_file)[1] == '.msh'
    assert os.path.splitext(h5_file)[1] == '.h5'

    # Get the xml mesh
    xml_file = '.'.join([root, 'xml'])
    subprocess.call(['dolfin-convert %s %s' % (msh_file, xml_file)], shell=True)
    # Success?
    assert os.path.exists(xml_file)

    mesh = Mesh(xml_file)
    out = HDF5File(mesh.mpi_comm(), h5_file, 'w')
    out.write(mesh, 'mesh')
                           
    for region in ('facet_region.xml', ):
        name, _ = region.split('_')
        r_xml_file = '_'.join([root, region])
        
        f = MeshFunction('size_t', mesh, r_xml_file)
        out.write(f, name)

    # Sucess?
    assert os.path.exists(h5_file)

    return mesh
    

def cleanup(files=None, exts=()):
    '''Get rid of xml'''
    if files is not None:
        return map(os.remove, files)
    else:
        files = filter(lambda f: any(map(f.endswith, exts)), os.listdir('.'))
        print 'Removing', files
        return cleanup(files)
                    
# --------------------------------------------------------------------

if __name__ == '__main__':
    from dolfin import Mesh, MeshFunction, HDF5File, mpi_comm_world, File
    import argparse

    parser = argparse.ArgumentParser(description='Convert msh file to h5')
    # If two args the other is 
    parser.add_argument('input', type=str, help='input msh file')
    # Output
    parser.add_argument('-output', type=str, help='Optional output', default='')
    # Save marking gunctions for checking
    save_parser = parser.add_mutually_exclusive_group(required=False)
    save_parser.add_argument('--save', dest='save', action='store_true')
    save_parser.add_argument('--no-save', dest='save', action='store_false')

    parser.set_defaults(save=True)
    # Remove Xml and stuff
    parser.add_argument('--cleanup', type=str, nargs='+',
                        help='extensions to delete', default=('.xml', ))
    args = parser.parse_args()

    # Protecting self
    assert not(set(('geo', '.geo')) & set(args.cleanup))
    

    if not args.output:
        args.output = '.'.join([os.path.splitext(args.input)[0], 'h5'])
    mesh = convert(args.input, args.output)

    if args.save:
        h5 = HDF5File(mpi_comm_world(), args.output, 'r')

        surfaces = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
        h5.read(surfaces, 'facet')

        File('results/%s_surf.pvd' % os.path.splitext(args.input)[0]) << surfaces

    cleanup(exts=args.cleanup)

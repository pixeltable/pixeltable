import * as pxt from '@pixeltable/sdk';

const TableModel = pxt.modelBase();

export class Notes extends TableModel('notes') {
  title = pxt.string();
  title_upper = this.title.upper();
}

export const addNote = pxt.insertRoute(Notes, {
  service: 'notes',
  path: '/notes',
  inputs: ['title'],
  outputs: ['title_upper'],
});
